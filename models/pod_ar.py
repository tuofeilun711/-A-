"""POD + AR reduced-order model for flow prediction.

Spatial basis from POD (SVD), temporal evolution from a ridge-regularised AR(p)
on the modal coefficients. predict_ensemble adds residual bootstrap samples for
an uncertainty estimate; predict is the deterministic rollout.
"""
import numpy as np

from data.h5_loader import build_snapshot_matrix


class PODFlowPredictor:
    def __init__(self, n_modes=50, ar_order=3, ridge_alpha=3.0):
        self.n_modes = n_modes
        self.ar_order = ar_order
        self.ridge_alpha = ridge_alpha
        self.U = None
        self.s = None
        self.mean_snapshot = None
        self.Ar_coeffs = None
        self.fluid_indices = None
        self.pod_energy = None
        self.r2 = None
        self.one_step_r2 = None
        self.ar_residuals = None

    def fit(self, train_fields, fluid_mask):
        if isinstance(train_fields, (list, tuple)):
            fields = [np.asarray(f) for f in train_fields]
        else:
            fields = [np.asarray(train_fields)]
        train_fields = np.concatenate(fields, axis=0)
        mask_2ch = np.stack([fluid_mask, fluid_mask], axis=0).reshape(-1)
        self.fluid_indices = np.where(mask_2ch)[0]

        snapshots = build_snapshot_matrix(train_fields, self.fluid_indices)
        self.mean_snapshot = snapshots.mean(axis=0, keepdims=True)
        X = snapshots - self.mean_snapshot
        N, M = X.shape

        if N < M:
            eigvals, eigvecs = np.linalg.eigh(X @ X.T)
            idx = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[idx], 0)
            eigvecs = eigvecs[:, idx]
            k = min(self.n_modes, N - 1)
            eigvals, eigvecs = eigvals[:k], eigvecs[:, :k]
            self.s = np.sqrt(eigvals)
            self.U = (X.T @ eigvecs) / (self.s + 1e-12)
            self.n_modes = k
        else:
            U_full, s_full, _ = np.linalg.svd(X, full_matrices=False)
            self.n_modes = min(self.n_modes, len(s_full))
            self.U = U_full[:, :self.n_modes]
            self.s = s_full[:self.n_modes]

        self.pod_energy = float(100 * np.sum(self.s ** 2) / np.sum(X ** 2))
        print("POD: %d modes, energy=%.1f%%" % (self.n_modes, self.pod_energy))

        modal_sequences = [
            (build_snapshot_matrix(field, self.fluid_indices) - self.mean_snapshot) @ self.U
            for field in fields
        ]
        self._fit_ar(modal_sequences)
        return self

    def _fit_ar(self, sequences):
        """Pool AR samples from trajectories without creating cross-boundary pairs."""
        if isinstance(sequences, np.ndarray):
            sequences = [sequences]
        requested_order = self.ar_order
        self.ar_order = min(
            requested_order,
            max(1, min(len(sequence) for sequence in sequences) // 2),
        )
        order = self.ar_order
        x_parts, y_parts = [], []
        for coeffs in sequences:
            if len(coeffs) <= order:
                continue
            x_parts.append(
                np.array(
                    [coeffs[t - order:t][::-1].flatten() for t in range(order, len(coeffs))]
                )
            )
            y_parts.append(coeffs[order:])
        if not x_parts:
            raise ValueError("Not enough training frames to fit the AR model")
        X_ar = np.concatenate(x_parts, axis=0)
        y_ar = np.concatenate(y_parts, axis=0)

        XtX = X_ar.T @ X_ar
        self.Ar_coeffs = np.linalg.solve(
            XtX + self.ridge_alpha * np.eye(XtX.shape[0]), X_ar.T @ y_ar).T
        resid = y_ar - X_ar @ self.Ar_coeffs.T
        self.ar_residuals = resid

        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y_ar - y_ar.mean(0)) ** 2) + 1e-12
        self.r2 = float(1 - ss_res / ss_tot)
        self.one_step_r2 = self.r2
        print("AR(%d): R^2=%.4f" % (order, self.r2))

    def _project(self, input_frames):
        flat = input_frames.reshape(input_frames.shape[0], -1)[:, self.fluid_indices]
        return list((flat - self.mean_snapshot) @ self.U)

    def _reconstruct(self, coeffs, H, W):
        fields = np.stack([(self.mean_snapshot + c @ self.U.T).ravel() for c in coeffs])
        full = np.zeros((len(coeffs), 2, H, W), dtype=np.float32)
        ic, ih, iw = np.unravel_index(self.fluid_indices, (2, H, W))
        full[:, ic, ih, iw] = fields
        return full

    def _rollout(self, input_frames, n_steps, rng=None):
        H, W = input_frames.shape[-2:]
        hist = self._project(input_frames)
        out = []
        for _ in range(n_steps):
            nxt = self.Ar_coeffs @ np.concatenate(hist[-self.ar_order:][::-1])
            if rng is not None and self.ar_residuals is not None:
                nxt = nxt + self.ar_residuals[rng.integers(len(self.ar_residuals))]
            out.append(nxt)
            hist.append(nxt)
        return self._reconstruct(out, H, W)

    def predict(self, input_frames, n_steps):
        """Deterministic rollout, returns (n_steps, 2, H, W) in normalized units."""
        return self._rollout(input_frames, n_steps)

    def predict_ensemble(self, input_frames, n_steps, n_members=20, seed=0):
        """Residual-bootstrap ensemble, returns (n_members, n_steps, 2, H, W)."""
        rng = np.random.default_rng(seed)
        return np.stack([self._rollout(input_frames, n_steps, rng) for _ in range(n_members)])

    def spectral_radius(self):
        """Largest eigenvalue magnitude of the AR companion matrix."""
        K, order = self.n_modes, self.ar_order
        C = np.zeros((order * K, order * K))
        for i in range(order):
            C[:K, i * K:(i + 1) * K] = self.Ar_coeffs[:, i * K:(i + 1) * K]
        for i in range(1, order):
            C[i * K:(i + 1) * K, (i - 1) * K:i * K] = np.eye(K)
        return float(np.max(np.abs(np.linalg.eigvals(C))))
