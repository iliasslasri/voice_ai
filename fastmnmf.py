import torch
import matplotlib.pyplot as plt


def circulant_matrix(n, m, eps=1e-2, dtype=torch.float32, device='cpu'):
	mat = torch.full((n, m), eps, dtype=dtype, device=device)
	if n < m:
		for i in range(n):
			for j in range(m // n):
				mat[i, i + j * n] = 1.0
		for i in range(m % n):
			mat[i, i + n * (m // n)] = 1.0
	else:
		for j in range(m):
			for i in range(n // m):
				mat[j + m * i, j] = 1.0
		for j in range(n % m):
			mat[j + m * (n // m), j] = 1.0
	return mat


class FastMNMF1:
    """
    The blind souce separation using FastMNMF1
    inputs:
        X: the observed complex spectrogram
        n_src: the number of sources
        n_bases : the hidden dimension in MNMF decomposition
        init_SCM: the method to initialize the spatial model ('circular', 'random', 'identity' or 'gradual')
        seed: set a random seed
        device: sent model to device

    inner_matrix:
        Q_FMM: diagonalizer that converts SCMs to diagonal matrices
        G_NFM: diagonal elements of the diagonalized SCMs
        W_NFK: basis vectors
        H_NKT: activations
        PSD_FTN: power spectral densities
        Qx_power_FTM: power spectra of Q_FMM times X_FTM
        Y_FTM: sum of (PSD_NFT x G_NFM) over all sources
    """
    def __init__(self, X, n_src, n_bases, n_iter, eps=1e-8, device=None, init_SCM = "rc", seed=None):

        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.X_FTM = X.to(device = self.device)

        self.dtype_complex = self.X_FTM.dtype
        self.dtype_real = self.X_FTM.real.dtype

        self.F, self.T, self.M = self.X_FTM.shape
        self.n_src = n_src
        self.n_bases = n_bases
        self.n_iterations = n_iter
        self.eps = eps
        self.g_eps = 5e-2

        self.metrics1 = []
        self.metrics2 = []
        self.iters = []

        self.init_SCM = init_SCM

        # X_rank1 should NOT be modified
        self.X_rank1 = torch.einsum('ftm, ftn -> ftmn', self.X_FTM, self.X_FTM.conj())

        # Initialize source and spatial models
        self.init_source_model(n_bases=n_bases)
        self.init_spatial_model()
        self.normalize()

        # Intermediate matrices that we should keep updated
        self.Qx_FTM = torch.einsum('fMm, ftm -> ftM', self.Q_FMM, self.X_FTM)
        self.X_hat = torch.abs(self.Qx_FTM)**2 
        self.Y_hat = torch.einsum('ftn, nfm -> ftm', self.PSD_FTN, self.G_NFM)
    
    def init_source_model(self, n_bases):
        # Spectral model : PSD of source n = WnHn
        self.W_NFK = torch.rand((self.n_src, self.F, n_bases), dtype=self.dtype_real, device=self.device)
        self.H_NKT = torch.rand((self.n_src, n_bases, self.T), dtype=self.dtype_real, device=self.device)
        self.PSD_FTN = torch.einsum('nfk,nkt -> ftn', self.W_NFK, self.H_NKT)

    def init_spatial_model(self):
        # Spatial model : SCM of source
        self.Q_FMM = torch.eye(self.M, dtype=self.dtype_complex, device=self.device).expand(self.F, -1, -1).clone().to(self.device)

        if self.init_SCM == "rc":
            vect_speech = torch.zeros((1,self.M), dtype=self.dtype_real, device=self.device)
            vect_speech[0,-1] = 1
            vect_noise = torch.ones((self.n_src-1, self.M), dtype=self.dtype_real, device=self.device)
            G_NM = torch.concat((vect_speech, vect_noise))
            self.G_NFM = G_NM.expand(self.F, self.n_src, self.M).permute(1,0,2)

            # Init Q
            Q_FMM_inv = torch.zeros((self.F, self.M, self.M), dtype=self.dtype_complex, device=self.device)
            X_rank1_sum = self.X_rank1.sum(dim=1)
            _, eigvecs = torch.linalg.eigh(X_rank1_sum)
            Q_FMM_inv = eigvecs
            self.Q_FMM = torch.linalg.inv(Q_FMM_inv)

        elif self.init_SCM == "gradual":
            self.G_NFM = (
                circulant_matrix(self.n_src, self.M, eps=self.g_eps, dtype=self.dtype_real, device=self.device).expand(self.F, self.n_src, self.M).permute(1,0,2).clone()
            )
            self.init_source_model(n_bases=2)
        elif self.init_SCM == "circular":
            self.G_NFM = (
                circulant_matrix(self.n_src, self.M, eps=self.g_eps, dtype=self.dtype_real, device=self.device).expand(self.F, self.n_src, self.M).permute(1,0,2).clone()
            )
        elif self.init_SCM == "random":
            self.G_NFM = torch.rand((self.n_src, self.F, self.M), device=self.device, dtype=self.dtype_real) + self.eps
        elif self.init_SCM == "identity":
            self.G_NFM = torch.ones((self.n_src, self.F, self.M), device=self.device, dtype=self.dtype_real) * self.g_eps
            for nm in range(min(self.n_src, self.M)):
                self.G_NFM[nm, :, nm] = 1.0
        else:
            raise ValueError(f"Unknown init_SCM value: {self.init_SCM}. Choose from 'gradual', 'circular', 'random', 'identity'.")

    def get_SCM(self, src, freq):
        Q_MM_inv = torch.linalg.inv(self.Q_FMM[freq])
        return torch.einsum('ij,jk,kl->il', Q_MM_inv, torch.diag(self.G_NFM[src,freq]).to(self.dtype_complex), Q_MM_inv.mH )
    
    def get_rank_SCM(self, freq=100):
        SCM_0 = self.get_SCM(src=0, freq=freq)
        SCM_1 = self.get_SCM(src=1, freq=freq)
        rank_0 = torch.linalg.matrix_rank(SCM_0)
        rank_1 = torch.linalg.matrix_rank(SCM_1)
        # print(f'rank_0 : {rank_0}')
        # print(f'rank_1 : {rank_1}')

    def compute_Y_hat(self):
        self.Y_hat = torch.einsum('ftn, nfm -> ftm', self.PSD_FTN, self.G_NFM)

    def compute_PSD_FTN(self):
        self.PSD_FTN = torch.einsum('nfk,nkt -> ftn', self.W_NFK, self.H_NKT)
        self.compute_Y_hat()

    def compute_Qx_FTM(self):
        self.Qx_FTM = torch.einsum('fMm, ftm -> ftM', self.Q_FMM, self.X_FTM)
        self.X_hat = torch.abs(self.Qx_FTM)**2 

    def update_WH(self):
        num_w = torch.einsum('nkt, nfm, ftm, ftm -> nfk', self.H_NKT, self.G_NFM, self.X_hat, self.Y_hat.pow(-2))
        denum_w = torch.einsum('nkt, nfm, ftm-> nfk', self.H_NKT, self.G_NFM, self.Y_hat.pow(-1))
        coeff_w = torch.sqrt(num_w/denum_w)
        assert torch.all(coeff_w >= 0), f"coeff_w contains negative values : {coeff_w}"
        self.W_NFK *= coeff_w
        self.W_NFK += self.eps
        self.compute_PSD_FTN()

        num_h = torch.einsum('nfk, nfm, ftm, ftm -> nkt', self.W_NFK, self.G_NFM, self.X_hat, self.Y_hat.pow(-2))
        denum_h = torch.einsum('nfk, nfm, ftm-> nkt', self.W_NFK, self.G_NFM, self.Y_hat.pow(-1))
        coeff_h = torch.sqrt(num_h/denum_h)
        assert torch.all(coeff_h >= 0), f"coeff_h contains negative values : {coeff_h}"
        self.H_NKT *= coeff_h
        self.H_NKT += self.eps
        self.compute_PSD_FTN()
    
    def update_G_NFM(self):
        num_g = torch.einsum('ftn, ftm, ftm -> nfm', self.PSD_FTN, self.X_hat, self.Y_hat.pow(-2))
        denum_g = torch.einsum('ftn, ftm-> nfm', self.PSD_FTN, self.Y_hat.pow(-1))
        coeff_g = torch.sqrt(num_g/denum_g)
        assert torch.all(coeff_g >= 0), f"coeff_g contains negative values : {coeff_g}"
        self.G_NFM *= coeff_g
        self.G_NFM += self.eps
        self.compute_Y_hat()

        self.test_parameters()
    
    def update_Q_IP(self):
        for m in range(self.M):
            V_FMM = (
                torch.einsum("ftij, ft -> fij", self.X_rank1, (1 / self.Y_hat[..., m]).to(self.dtype_complex))
                / self.T
            )
            tmp_FM = torch.linalg.solve(
                self.Q_FMM @ V_FMM, torch.eye(self.M, dtype=self.dtype_complex, device=self.device)[None]
            )[..., m]
            self.Q_FMM[:, m] = (
                tmp_FM / torch.sqrt(torch.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM))[:, None]
            ).conj()
        
        self.compute_Qx_FTM()
        self.test_parameters()

    def normalize(self):        
        mu = torch.einsum('fmn, fmn -> fm', self.Q_FMM, self.Q_FMM.conj()).real + self.eps
        assert torch.all(mu >= 0), f"Some values of mu are negative : {mu}"
        self.Q_FMM = torch.einsum("fm, fmM -> fmM", mu.pow(-1/2), self.Q_FMM)
        self.G_NFM = torch.einsum("fm, nfm -> nfm", mu.pow(-1), self.G_NFM)

        phi = self.G_NFM.sum(dim=2) + self.eps
        assert torch.all(phi >= 0), f"Some values of phi are negative : {phi}"
        self.G_NFM = torch.einsum("nf, nfm -> nfm", phi.pow(-1), self.G_NFM)
        self.W_NFK = torch.einsum("nf, nfk -> nfk", phi, self.W_NFK)

        nu = self.W_NFK.sum(dim=1) + self.eps
        assert torch.all(nu >= 0), f"Some values of nu are negative : {nu}"
        self.W_NFK = torch.einsum("nk, nfk -> nfk", nu.pow(-1), self.W_NFK)
        self.H_NKT = torch.einsum("nk, nkt -> nkt", nu, self.H_NKT)

        # Update intermediate matrices for the next loop
        self.compute_PSD_FTN()
        self.compute_Qx_FTM()

        self.test_parameters()


    def fit(self, callback=0,
            metric_name1='SI-SDR',
            metric_name2=None,
            reference_source=None,
            noise=None, n_fft=None,
            hop_size=None, length=None,
            src=0, norm_factor=1,
            spec_back=False):
        # X is the input data of shape (F, T, M)
        for iter in range(self.n_iterations):

            if self.init_SCM == "gradual" and iter == 50:
                self.init_source_model(n_bases=self.n_bases)
            self.update_WH()
            self.update_G_NFM()
            self.update_Q_IP()
            if iter%1 == 0:
                self.normalize()

            # Plotting
        #     if callback != 0 and iter % callback == 0:
        #         self.iters.append(iter+1)
        #         if metric_name1 == 'LLH' and metric_name2 is None:
        #             self.metrics1.append(self.calculate_log_likelihood().item())
        #         else:
        #             assert self.n_src == 2, f'Error, plotting the sdr is available for 2 src here n_src : {self.n_src}'
        #             assert reference_source is not None and noise is not None and n_fft is not None and hop_size is not None and length is not None
        #             metrics = self.compute_metrics(reference_source=reference_source, 
        #                                            noise=noise, n_fft=n_fft, 
        #                                            hop_size=hop_size, 
        #                                            length=length,
        #                                            norm_factor=norm_factor,
        #                                            spec_back=spec_back,
        #                                            src=src)
        #             if metric_name1 == 'LLH':
        #                 self.metrics1.append(self.calculate_log_likelihood().item())
        #             else:
        #                 self.metrics1.append(round(metrics[metric_name1],2))
        #             if metric_name2 is not None:
        #                 self.metrics2.append(round(metrics[metric_name2],2))

        # if self.metrics1 != []:
        #     self.plot_metrics(metric_name1=metric_name1,
        #                       metric_name2=metric_name2)

        # print(f"Model has been fit with number of iterations : {self.n_iterations}")
        # print(f"Device is : {self.device}")

    def separated(self, mic_index=0):
        Y_NFTM = torch.einsum("ftn, nfm -> nftm", self.PSD_FTN, self.G_NFM).to(self.dtype_complex)
        Y_FTM = Y_NFTM.sum(axis=0)
        Qinv_FMM = torch.linalg.solve(
            self.Q_FMM, torch.eye(self.M, dtype=self.dtype_complex, device=self.device)[None]
        )

        if mic_index is None:
            self.separated_spec = torch.einsum(
                "fmj, ftj, nftj -> mnft", Qinv_FMM, self.Qx_FTM / Y_FTM, Y_NFTM
            )
        else:
            self.separated_spec = torch.einsum(
                "fj, ftj, nftj -> nft", Qinv_FMM[:, mic_index], self.Qx_FTM / Y_FTM, Y_NFTM
            )
        return self.separated_spec
    
    def calculate_log_likelihood(self):
        log_likelihood = (
            -(self.X_hat / self.Y_hat + torch.log(self.Y_hat)).sum()
            + self.T * (torch.log(torch.linalg.det(torch.einsum('fij,fkj->fik', self.Q_FMM, self.Q_FMM.conj()).real))).sum()
        )
        return log_likelihood
    
    def plot_metrics(self, metric_name1, metric_name2):
        """
        Plot the log metrics1 evolution throughout iterations
        """
        iters = self.iters 
        metrics1 = self.metrics1
        metrics2 = self.metrics2

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Axe Y gauche : metrics1
        l1, = ax1.plot(iters, metrics1, '-', color='blue', label=metric_name1)
        s1 = ax1.scatter(iters, metrics1, marker='^', color='red', s=50)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel(metric_name1, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--')

        # condition sur les bornes Y de la log-metrics1
        if len(metrics1) > 1 and metric_name1 == 'LLH':
            ymin = min(metrics1[1:]) * 0.95
            ymax = max(metrics1) * 1.05
            ax1.set_ylim(ymin, ymax)

        # Axe Y droit : Metrics (ex. SI-SDR)
        if len(metrics2) == len(iters):
            ax2 = ax1.twinx()
            l2, = ax2.plot(iters, metrics2, '-', color='green', label=metric_name1)
            s2 = ax2.scatter(iters, metrics2, marker='o', color='orange', s=50)
            ax2.set_ylabel(metric_name2, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            plt.title(metric_name1 + ' & ' + metric_name2)
        else:
            plt.title(metric_name1)

        fig.tight_layout()
        plt.show()

        
    
    def test_parameters(self):
        # Test the shape of the parameters
        # assert self.W_NFK.shape == (self.n_src, self.F, self.n_bases), f"Shape of W is not correct : {self.W_NFK.shape}"
        # assert self.H_NKT.shape == (self.n_src, self.n_bases, self.T), f"Shape of H is not correct : {self.H_NKT.shape}"
        assert self.G_NFM.shape == (self.n_src, self.F, self.M), f"Shape of G is not correct : {self.G_NFM.shape}"
        assert self.Q_FMM.shape == (self.F, self.M, self.M), f"Shape of Q is not correct : {self.Q_FMM.shape}"
        assert self.X_FTM.shape == (self.F, self.T, self.M), f"Shape of X is not correct : {self.X_FTM.shape}"
        assert self.X_hat.shape == (self.F, self.T, self.M), f"Shape of X_hat is not correct : {self.X_hat.shape}"
        assert self.Y_hat.shape == (self.F, self.T, self.M), f"Shape of Y_hat is not correct : {self.Y_hat.shape}"
        
        # Test the dtype of the parameters
        assert self.W_NFK.dtype == self.dtype_real, f"Type of W is not correct : {self.W_NFK.dtype}"
        assert self.H_NKT.dtype == self.dtype_real, f"Type of H is not correct : {self.H_NKT.dtype}"
        assert self.G_NFM.dtype == self.dtype_real, f"Type of G is not correct : {self.G_NFM.dtype}"
        assert self.X_hat.dtype == self.dtype_real, f"Type of G is not correct : {self.X_hat.dtype}"
        assert self.Y_hat.dtype == self.dtype_real, f"Type of G is not correct : {self.Y_hat.dtype}"
        assert self.Qx_FTM.dtype == self.dtype_complex, f"Type of Qx is not correct : {self.Qx_FTM.dtype}"
        assert self.Q_FMM.dtype == self.dtype_complex, f"Type of Q is not correct : {self.Q_FMM.dtype}"

        # Test the postivity of the parameters
        assert torch.all(self.W_NFK >= 0), f"Some values of W are negative : {self.W_NFK}"
        assert torch.all(self.H_NKT >= 0), f"Some values of H are negative : {self.H_NKT}"
        assert torch.all(self.G_NFM >= 0), f"Some values of G are negative : {self.G_NFM}"
        assert torch.all(self.X_hat >= 0), f"Some values of X_hat are negative or nan "
        assert torch.all(self.Y_hat >= 0), f"Some values of Y_hat are negative : {self.Y_hat}"


#######################################################################################################


class FastMNMF2:
    """
    The blind souce separation using FastMNMF2
    inputs:
        X: the observed complex spectrogram
        n_src: the number of sources
        n_bases : the hidden dimension in MNMF decomposition
        init_SCM: the method to initialize the spatial model ('circular', 'random', 'identity' or 'gradual')
        seed: set a random seed
        device: sent model to device

    inner_matrix:
        Q_FMM: diagonalizer that converts SCMs to diagonal matrices
        G_NM: diagonal elements of the diagonalized SCMs
        W_NFK: basis vectors
        H_NKT: activations
        PSD_FTN: power spectral densities
        Qx_power_FTM: power spectra of Q_FMM times X_FTM
        Y_FTM: sum of (PSD_NFT x G_NM) over all sources
    """
    def __init__(self, X, n_src, n_bases, n_iter, eps=1e-8, device=None, init_SCM = "rc", seed=None):

        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.X_FTM = X.to(device = self.device)

        self.dtype_complex = self.X_FTM.dtype
        self.dtype_real = self.X_FTM.real.dtype

        self.F, self.T, self.M = self.X_FTM.shape
        self.n_src = n_src
        self.n_bases = n_bases
        self.n_iterations = n_iter
        self.eps = eps
        self.g_eps = 5e-2

        self.metrics1 = []
        self.metrics2 = []
        self.iters = []

        self.init_SCM = init_SCM

        # X_rank1 should NOT be modified
        self.X_rank1 = torch.einsum('ftm, ftn -> ftmn', self.X_FTM, self.X_FTM.conj())

        # Initialize source and spatial models
        self.init_source_model(n_bases=n_bases)
        self.init_spatial_model()
        self.normalize()

        # Intermediate matrices that we should keep updated
        self.Qx_FTM = torch.einsum('fMm, ftm -> ftM', self.Q_FMM, self.X_FTM)
        self.X_hat = torch.abs(self.Qx_FTM)**2 
        self.Y_hat = torch.einsum('ftn, nm -> ftm', self.PSD_FTN, self.G_NM)
    
    def init_source_model(self, n_bases):
        # Spectral model : PSD of source n = WnHn
        self.W_NFK = torch.rand((self.n_src, self.F, n_bases), dtype=self.dtype_real, device=self.device)
        self.H_NKT = torch.rand((self.n_src, n_bases, self.T), dtype=self.dtype_real, device=self.device)
        self.PSD_FTN = torch.einsum('nfk,nkt -> ftn', self.W_NFK, self.H_NKT)

    def init_spatial_model(self):
        # Spatial model : SCM of source
        self.Q_FMM = torch.eye(self.M, dtype=self.dtype_complex, device=self.device).expand(self.F, -1, -1).clone().to(self.device)

        if self.init_SCM == "rc":
            vect_speech = torch.zeros((1,self.M), dtype=self.dtype_real, device=self.device)
            vect_speech[0,-1] = 1
            vect_noise = torch.ones((self.n_src-1, self.M), dtype=self.dtype_real, device=self.device)
            self.G_NM = torch.concat((vect_speech, vect_noise))

            # Init Q
            Q_FMM_inv = torch.zeros((self.F, self.M, self.M), dtype=self.dtype_complex, device=self.device)
            X_rank1_sum = self.X_rank1.sum(dim=1)
            _, eigvecs = torch.linalg.eigh(X_rank1_sum)
            Q_FMM_inv = eigvecs
            self.Q_FMM = torch.linalg.inv(Q_FMM_inv)

        elif self.init_SCM == "gradual":
            self.G_NM = circulant_matrix(self.n_src, self.M, eps=self.g_eps, dtype=self.dtype_real, device=self.device).clone()
            self.init_source_model(n_bases=2)
        elif self.init_SCM == "circular":
            self.G_NM = circulant_matrix(self.n_src, self.M, eps=self.g_eps, dtype=self.dtype_real, device=self.device).clone()
        elif self.init_SCM == "random":
            self.G_NM = torch.rand((self.n_src, self.M), device=self.device, dtype=self.dtype_real) + self.eps
        elif self.init_SCM == "identity":
            self.G_NM = torch.ones((self.n_src, self.M), device=self.device, dtype=self.dtype_real) * self.g_eps
            for nm in range(min(self.n_src, self.M)):
                self.G_NM[nm, nm] = 1.0
        else:
            raise ValueError(f"Unknown init_SCM value: {self.init_SCM}. Choose from 'gradual', 'circular', 'random', 'identity'.")

    def get_SCM(self, src, freq):
        Q_MM_inv = torch.linalg.inv(self.Q_FMM[freq])
        return torch.einsum('ij,jk,kl->il', Q_MM_inv, torch.diag(self.G_NM[src]).to(self.dtype_complex), Q_MM_inv.mH )
    
    def get_rank_SCM(self, freq=100):
        SCM_0 = self.get_SCM(src=0, freq=freq)
        SCM_1 = self.get_SCM(src=1, freq=freq)
        rank_0 = torch.linalg.matrix_rank(SCM_0)
        rank_1 = torch.linalg.matrix_rank(SCM_1)
        # print(f'rank_0 : {rank_0}')
        # print(f'rank_1 : {rank_1}')

    def compute_Y_hat(self):
        self.Y_hat = torch.einsum('ftn, nm -> ftm', self.PSD_FTN, self.G_NM)

    def compute_PSD_FTN(self):
        self.PSD_FTN = torch.einsum('nfk,nkt -> ftn', self.W_NFK, self.H_NKT)
        self.compute_Y_hat()

    def compute_Qx_FTM(self):
        self.Qx_FTM = torch.einsum('fMm, ftm -> ftM', self.Q_FMM, self.X_FTM)
        self.X_hat = torch.abs(self.Qx_FTM)**2 

    def update_WH(self):
        num_w = torch.einsum('nkt, nm, ftm, ftm -> nfk', self.H_NKT, self.G_NM, self.X_hat, self.Y_hat.pow(-2))
        denum_w = torch.einsum('nkt, nm, ftm-> nfk', self.H_NKT, self.G_NM, self.Y_hat.pow(-1))
        coeff_w = torch.sqrt(num_w/denum_w)
        assert torch.all(coeff_w >= 0), f"coeff_w contains negative values : {coeff_w}"
        self.W_NFK *= coeff_w
        self.W_NFK += self.eps
        self.compute_PSD_FTN()

        num_h = torch.einsum('nfk, nm, ftm, ftm -> nkt', self.W_NFK, self.G_NM, self.X_hat, self.Y_hat.pow(-2))
        denum_h = torch.einsum('nfk, nm, ftm-> nkt', self.W_NFK, self.G_NM, self.Y_hat.pow(-1))
        coeff_h = torch.sqrt(num_h/denum_h)
        assert torch.all(coeff_h >= 0), f"coeff_h contains negative values : {coeff_h}"
        self.H_NKT *= coeff_h
        self.H_NKT += self.eps
        self.compute_PSD_FTN()
    
    def update_G_NM(self):
        num_g = torch.einsum('ftn, ftm, ftm -> nm', self.PSD_FTN, self.X_hat, self.Y_hat.pow(-2))
        denum_g = torch.einsum('ftn, ftm-> nm', self.PSD_FTN, self.Y_hat.pow(-1))
        coeff_g = torch.sqrt(num_g/denum_g)
        assert torch.all(coeff_g >= 0), f"coeff_g contains negative values : {coeff_g}"
        self.G_NM *= coeff_g
        self.G_NM += self.eps
        self.compute_Y_hat()

        self.test_parameters()
    
    def update_Q_IP(self):
        for m in range(self.M):
            V_FMM = (
                torch.einsum("ftij, ft -> fij", self.X_rank1, (1 / self.Y_hat[..., m]).to(self.dtype_complex))
                / self.T
            )
            tmp_FM = torch.linalg.solve(
                self.Q_FMM @ V_FMM, torch.eye(self.M, dtype=self.dtype_complex, device=self.device)[None]
            )[..., m]
            self.Q_FMM[:, m] = (
                tmp_FM / torch.sqrt(torch.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM))[:, None]
            ).conj()
        
        self.compute_Qx_FTM()
        self.test_parameters()

    def normalize(self):
        mu = torch.einsum('fmM,fmM->f', self.Q_FMM, self.Q_FMM.conj()).real / self.M + self.eps
        assert torch.all(mu >= 0), f"Some values of mu are negative : {mu}"
        self.Q_FMM = torch.einsum("f, fmM -> fmM", mu.pow(-1/2), self.Q_FMM)
        self.W_NFK = torch.einsum("f, nfk -> nfk", mu.pow(-1), self.W_NFK)

        phi = self.G_NM.sum(dim=1) + self.eps
        assert torch.all(phi >= 0), f"Some values of phi are negative : {phi}"
        self.G_NM = torch.einsum("n, nm -> nm", phi.pow(-1), self.G_NM)
        self.W_NFK = torch.einsum("n, nfk -> nfk", phi, self.W_NFK)

        nu = self.W_NFK.sum(dim=1) + self.eps
        assert torch.all(nu >= 0), f"Some values of nu are negative : {nu}"
        self.W_NFK = torch.einsum("nk, nfk -> nfk", nu.pow(-1), self.W_NFK)
        self.H_NKT = torch.einsum("nk, nkt -> nkt", nu, self.H_NKT)

        # Update intermediate matrices for the next loop
        self.compute_PSD_FTN()
        self.compute_Qx_FTM()

        self.test_parameters()

    def fit(self, callback=0,
            metric_name1='SI-SDR',
            metric_name2=None,
            reference_source=None,
            noise=None, n_fft=None,
            hop_size=None, length=None,
            src=0, norm_factor=1,
            spec_back=False):
        # X is the input data of shape (F, T, M)
        for iter in range(self.n_iterations):

            if self.init_SCM == "gradual" and iter == 50:
                self.init_source_model(n_bases=self.n_bases)
            self.update_WH()
            self.update_G_NM()
            self.update_Q_IP()
            if iter%1 == 0:
                self.normalize()

        #     # Plotting
        #     if callback != 0 and iter % callback == 0:
        #         self.iters.append(iter+1)
        #         if metric_name1 == 'LLH' and metric_name2 is None:
        #             self.metrics1.append(self.calculate_log_likelihood().item())
        #         else:
        #             assert self.n_src == 2, f'Error, plotting the sdr is available for 2 src here n_src : {self.n_src}'
        #             assert reference_source is not None and noise is not None and n_fft is not None and hop_size is not None and length is not None
        #             metrics = self.compute_metrics(reference_source=reference_source, 
        #                                            noise=noise, n_fft=n_fft, 
        #                                            hop_size=hop_size, 
        #                                            length=length,
        #                                            norm_factor=norm_factor,
        #                                            spec_back=spec_back,
        #                                            src=src)
        #             if metric_name1 == 'LLH':
        #                 self.metrics1.append(self.calculate_log_likelihood().item())
        #             else:
        #                 self.metrics1.append(round(metrics[metric_name1],2))
        #             if metric_name2 is not None:
        #                 self.metrics2.append(round(metrics[metric_name2],2))

        # if self.metrics1 != []:
        #     self.plot_metrics(metric_name1=metric_name1,
        #                       metric_name2=metric_name2)

        # print(f"Model has been fit with number of iterations : {self.n_iterations}")
        # print(f"Device is : {self.device}")

    def separated(self, mic_index=0):
        Y_NFTM = torch.einsum("ftn, nm -> nftm", self.PSD_FTN, self.G_NM).to(self.dtype_complex)
        Y_FTM = Y_NFTM.sum(axis=0)
        Qinv_FMM = torch.linalg.solve(
            self.Q_FMM, torch.eye(self.M, dtype=self.dtype_complex, device=self.device)[None]
        )

        if mic_index is None:
            self.separated_spec = torch.einsum(
                "fmj, ftj, nftj -> mnft", Qinv_FMM, self.Qx_FTM / Y_FTM, Y_NFTM
            )
        else:
            self.separated_spec = torch.einsum(
                "fj, ftj, nftj -> nft", Qinv_FMM[:, mic_index], self.Qx_FTM / Y_FTM, Y_NFTM
            )
        return self.separated_spec
    
    def calculate_log_likelihood(self):
        log_likelihood = (
            -(self.X_hat / self.Y_hat + torch.log(self.Y_hat)).sum()
            + self.T * (torch.log(torch.linalg.det(torch.einsum('fij,fkj->fik', self.Q_FMM, self.Q_FMM.conj()).real))).sum()
        )
        return log_likelihood
    
    def plot_metrics(self, metric_name1, metric_name2):
        """
        Plot the log metrics1 evolution throughout iterations
        """
        iters = self.iters 
        metrics1 = self.metrics1
        metrics2 = self.metrics2

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Axe Y gauche : Log-metrics1
        l1, = ax1.plot(iters, metrics1, '-', color='blue', label=metric_name1)
        s1 = ax1.scatter(iters, metrics1, marker='^', color='red', s=50)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel(metric_name1, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--')

        # condition sur les bornes Y de la log-metrics1
        if len(metrics1) > 1 and metric_name1 == 'LLH':
            ymin = min(metrics1[1:]) * 0.95
            ymax = max(metrics1) * 1.05
            ax1.set_ylim(ymin, ymax)

        # Axe Y droit : Metrics (ex. SI-SDR)
        if len(metrics2) == len(iters):
            ax2 = ax1.twinx()
            l2, = ax2.plot(iters, metrics2, '-', color='green', label=metric_name2)
            s2 = ax2.scatter(iters, metrics2, marker='o', color='orange', s=50)
            ax2.set_ylabel(metric_name2, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            plt.title(metric_name2 + ' & ' + metric_name1)
        else:
            plt.title(metric_name2)

        fig.tight_layout()
        plt.show()


    # def compute_metrics(self, reference_source, noise, n_fft, hop_size, length, norm_factor, spec_back, ref_mic_idx=0, src=0):
    #     _, separated_src = get_separation(model=self,
    #                                       n_fft=n_fft,
    #                                       hop_size=hop_size,
    #                                       length=length,
    #                                       norm_factor=norm_factor,
    #                                       spec_reverse_transform=spec_back,
    #                                       ref_mic_idx=ref_mic_idx)    # n_src, n_samples

    #     separated_src = separated_src.cpu().numpy()
    #     metrics = separation_metrics(reference_source=reference_source,
    #                      estimated_source=separated_src[src],
    #                      noise=noise
    #                      )

    #     return metrics
    
    def test_parameters(self):
        # Test the shape of the parameters
        # assert self.W_NFK.shape == (self.n_src, self.F, self.n_bases), f"Shape of W is not correct : {self.W_NFK.shape}"
        # assert self.H_NKT.shape == (self.n_src, self.n_bases, self.T), f"Shape of H is not correct : {self.H_NKT.shape}"
        assert self.G_NM.shape == (self.n_src, self.M), f"Shape of G is not correct : {self.G_NM.shape}"
        assert self.Q_FMM.shape == (self.F, self.M, self.M), f"Shape of Q is not correct : {self.Q_FMM.shape}"
        assert self.X_FTM.shape == (self.F, self.T, self.M), f"Shape of X is not correct : {self.X_FTM.shape}"
        assert self.X_hat.shape == (self.F, self.T, self.M), f"Shape of X_hat is not correct : {self.X_hat.shape}"
        assert self.Y_hat.shape == (self.F, self.T, self.M), f"Shape of Y_hat is not correct : {self.Y_hat.shape}"
        
        # Test the dtype of the parameters
        assert self.W_NFK.dtype == self.dtype_real, f"Type of W is not correct : {self.W_NFK.dtype}"
        assert self.H_NKT.dtype == self.dtype_real, f"Type of H is not correct : {self.H_NKT.dtype}"
        assert self.G_NM.dtype == self.dtype_real, f"Type of G is not correct : {self.G_NM.dtype}"
        assert self.X_hat.dtype == self.dtype_real, f"Type of G is not correct : {self.X_hat.dtype}"
        assert self.Y_hat.dtype == self.dtype_real, f"Type of G is not correct : {self.Y_hat.dtype}"
        assert self.Qx_FTM.dtype == self.dtype_complex, f"Type of Qx is not correct : {self.Qx_FTM.dtype}"
        assert self.Q_FMM.dtype == self.dtype_complex, f"Type of Q is not correct : {self.Q_FMM.dtype}"

        # Test the postivity of the parameters
        assert torch.all(self.W_NFK >= 0), f"Some values of W are negative : {self.W_NFK}"
        assert torch.all(self.H_NKT >= 0), f"Some values of H are negative : {self.H_NKT}"
        assert torch.all(self.G_NM >= 0), f"Some values of G are negative : {self.G_NM}"
        assert torch.all(self.X_hat >= 0), f"Some values of X_hat are negative or nan "
        assert torch.all(self.Y_hat >= 0), f"Some values of Y_hat are negative : {self.Y_hat}"