"""Harmonisasi efek situs dengan ComBat (Johnson dkk., 2007; Fortin dkk., 2017/2018) versi fit/transform.

Berbeda dengan neuroCombat yang mengharmonisasi seluruh data sekaligus, kelas `ComBat` di sini
mempelajari parameter (efek situs gamma*, delta*, rata-rata, varians, efek kovariat) HANYA dari data
latih, lalu menerapkannya ke data uji. Dengan begitu harmonisasi tidak membocorkan informasi data uji.

Kovariat (mis. usia, jenis kelamin) dijaga agar efek biologisnya tidak ikut terhapus. Diagnosis
SENGAJA tidak dijadikan kovariat: label data uji tidak boleh dipakai saat transform.
Hasil `fit(X).transform(X)` pada data yang sama identik dengan neuroCombat (eb=True, parametric=True).
"""
import numpy as np

_EPS = 1e-8


class ComBat:
    def __init__(self, eb=True, tol=1e-4, max_iter=1000):
        self.eb, self.tol, self.max_iter = eb, tol, max_iter

    @staticmethod
    def _cov(covars, n):
        return np.zeros((n, 0)) if covars is None else np.asarray(covars, dtype=float).reshape(n, -1)

    def fit(self, X, batch, covars=None):
        """X: (subjek x fitur), batch: label situs per subjek, covars: (subjek x kovariat) numerik."""
        X = np.asarray(X, dtype=float)
        batch = np.asarray(batch).astype(str)
        n = X.shape[0]
        self.batches_ = np.unique(batch)
        B = (batch[:, None] == self.batches_[None, :]).astype(float)          # one-hot situs
        C = self._cov(covars, n)
        design = np.hstack([B, C])
        beta = np.linalg.lstsq(design, X, rcond=None)[0]                      # (k + p) x fitur
        k = len(self.batches_)
        n_per = B.sum(axis=0)
        self.grand_mean_ = (n_per / n) @ beta[:k]
        self.beta_cov_ = beta[k:]
        self.var_pooled_ = ((X - design @ beta) ** 2).mean(axis=0) + _EPS
        Z = (X - self.grand_mean_ - C @ self.beta_cov_) / np.sqrt(self.var_pooled_)

        gamma_hat = np.stack([Z[batch == b].mean(axis=0) for b in self.batches_])
        # Situs dengan < 2 subjek latih: efek situs tidak bisa dipisahkan dari ciri subjek itu sendiri,
        # sehingga situs tersebut TIDAK dikoreksi (gamma = 0, delta = 1) dan tidak ikut menentukan prior EB.
        cukup = n_per >= 2
        delta_hat = np.stack([Z[batch == b].var(axis=0, ddof=1) if c else np.ones(Z.shape[1])
                              for b, c in zip(self.batches_, cukup)]) + _EPS
        if not self.eb:
            self.gamma_ = np.where(cukup[:, None], gamma_hat, 0.0)
            self.delta_ = np.where(cukup[:, None], delta_hat, 1.0)
            return self

        # Empirical Bayes parametrik: prior normal untuk gamma, inverse-gamma untuk delta (per situs)
        gamma_bar = gamma_hat.mean(axis=1)
        t2 = gamma_hat.var(axis=1, ddof=1)
        m = delta_hat.mean(axis=1)
        s2 = delta_hat.var(axis=1, ddof=1)
        a_prior = (2 * s2 + m ** 2) / s2
        b_prior = (m * s2 + m ** 3) / s2
        self.gamma_ = np.empty_like(gamma_hat)
        self.delta_ = np.empty_like(delta_hat)
        for i, b in enumerate(self.batches_):
            if not cukup[i]:
                self.gamma_[i], self.delta_[i] = 0.0, 1.0
                continue
            Zb, nb = Z[batch == b], n_per[i]
            g_old, d_old = gamma_hat[i].copy(), delta_hat[i].copy()
            for _ in range(self.max_iter):
                g_new = (t2[i] * nb * gamma_hat[i] + d_old * gamma_bar[i]) / (t2[i] * nb + d_old)
                ssq = ((Zb - g_new) ** 2).sum(axis=0)
                d_new = (0.5 * ssq + b_prior[i]) / (nb / 2.0 + a_prior[i] - 1.0)
                selesai = max(np.max(np.abs(g_new - g_old) / np.abs(g_old).clip(_EPS)),
                              np.max(np.abs(d_new - d_old) / d_old)) < self.tol
                g_old, d_old = g_new, d_new
                if selesai:
                    break
            self.gamma_[i], self.delta_[i] = g_old, d_old
        return self

    def transform(self, X, batch, covars=None):
        X = np.asarray(X, dtype=float)
        batch = np.asarray(batch).astype(str)
        # Situs yang tidak ada di data latih (mis. situs sangat kecil yang seluruh subjeknya jatuh di data uji):
        # efek situsnya tidak diketahui, sehingga hanya distandardisasi tanpa koreksi situs (gamma = 0, delta = 1).
        dikenal = np.isin(batch, self.batches_)
        idx = np.searchsorted(self.batches_, np.where(dikenal, batch, self.batches_[0]))
        gamma = np.where(dikenal[:, None], self.gamma_[idx], 0.0)
        delta = np.where(dikenal[:, None], self.delta_[idx], 1.0)
        mean = self.grand_mean_ + self._cov(covars, X.shape[0]) @ self.beta_cov_
        Z = (X - mean) / np.sqrt(self.var_pooled_)
        Z = (Z - gamma) / np.sqrt(delta)
        return Z * np.sqrt(self.var_pooled_) + mean

    def fit_transform(self, X, batch, covars=None):
        return self.fit(X, batch, covars).transform(X, batch, covars)
