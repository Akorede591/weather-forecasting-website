import numpy as np
import warnings  # Added to manage terminal warnings cleanly
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress Convergence Warnings from the fast estimator to keep the terminal clean
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# =========================================================================
# PHASE 1: BINARY PARTICLE SWARM OPTIMIZATION (BPSO) FEATURE SELECTION
# =========================================================================
class BPSOFeatureSelection:
    def __init__(self, X, y, num_particles=10, max_iter=10, alpha=0.85):
        """
        Initializes the Binary PSO optimization engine.
        """
        self.X = X
        self.y = y
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.alpha = alpha
        self.num_features = X.shape[1]
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def _calculate_fitness(self, position):
        selected_indices = np.where(position == 1)[0]
        
        if len(selected_indices) == 0:
            return float('inf')
        
        X_tr_sub = self.X_train[:, selected_indices]
        X_val_sub = self.X_val[:, selected_indices]
        
        # INCREASED max_iter slightly to 100 for smoother intermediate fits
        evaluator = SGDRegressor(max_iter=100, random_state=42, tol=1e-3)
        try:
            evaluator.fit(X_tr_sub, self.y_train)
            mse = mean_squared_error(self.y_val, evaluator.predict(X_val_sub))
        except:
            mse = float('inf')
            
        feature_ratio = len(selected_indices) / self.num_features
        fitness_score = (self.alpha * mse) + ((1 - self.alpha) * feature_ratio)
        return fitness_score

    def optimize(self, progress_callback=None):
        velocities = np.random.uniform(-4, 4, (self.num_particles, self.num_features))
        positions = np.random.randint(2, size=(self.num_particles, self.num_features))
        
        p_best_positions = np.copy(positions)
        p_best_scores = np.array([self._calculate_fitness(p) for p in positions])
        
        g_best_idx = np.argmin(p_best_scores)
        g_best_position = np.copy(p_best_positions[g_best_idx])
        g_best_score = p_best_scores[g_best_idx]
        
        w, c1, c2 = 0.7, 1.5, 1.5
        
        for iteration in range(self.max_iter):
            r1 = np.random.rand(self.num_particles, self.num_features)
            r2 = np.random.rand(self.num_particles, self.num_features)
            
            cognitive = c1 * r1 * (p_best_positions - positions)
            social = c2 * r2 * (g_best_position - positions)
            velocities = w * velocities + cognitive + social
            velocities = np.clip(velocities, -4, 4)
            
            probabilities = self._sigmoid(velocities)
            positions = (np.random.rand(self.num_particles, self.num_features) < probabilities).astype(int)
            
            for i in range(self.num_particles):
                current_score = self._calculate_fitness(positions[i])
                if current_score < p_best_scores[i]:
                    p_best_scores[i] = current_score
                    p_best_positions[i] = np.copy(positions[i])
                    
            best_iter_idx = np.argmin(p_best_scores)
            if p_best_scores[best_iter_idx] < g_best_score:
                g_best_score = p_best_scores[best_iter_idx]
                g_best_position = np.copy(p_best_positions[best_iter_idx])
            
            if progress_callback:
                progress_callback(iteration + 1, self.max_iter, g_best_score)
            
        return g_best_position


# =========================================================================
# PHASE 2: PIPELINE EXECUTION ENGINE (BPSO + BPNN TRAINING LOOP)
# =========================================================================
def run_hybrid_pipeline(df, target_column, excluded_cols, swarm_size, pso_iters, bpnn_topology):
    working_df = df.drop(columns=[col for col in excluded_cols if col in df.columns], errors='ignore')
    clean_df = working_df.dropna()
    
    features_list = [col for col in clean_df.columns if col != target_column]
    X_raw = clean_df[features_list].select_dtypes(include=[np.number]).values
    y_raw = clean_df[target_column].values
    
    if X_raw.shape[1] == 0:
        raise ValueError("Dataset does not contain valid numeric attributes to process.")
        
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pso_engine = BPSOFeatureSelection(X_scaled, y_raw, num_particles=swarm_size, max_iter=pso_iters)
    optimal_mask = pso_engine.optimize()
    selected_features = [features_list[i] for i, mask in enumerate(optimal_mask) if mask == 1]
    
    if len(selected_features) == 0:
        raise RuntimeError("BPSO optimization failed to identify any relevant feature descriptors.")
        
    X_optimized = X_scaled[:, np.where(optimal_mask == 1)[0]]
    
    X_train, X_test, y_train, y_test = train_test_split(X_optimized, y_raw, test_size=0.2, random_state=42)
    X_train_full, X_test_full, _, _ = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42)
    
    bpnn_model = MLPRegressor(
        hidden_layer_sizes=bpnn_topology, 
        activation='relu', 
        solver='adam', 
        learning_rate_init=0.01, 
        max_iter=300, 
        random_state=42, 
        early_stopping=True
    )
    bpnn_model.fit(X_train, y_train)
    bpnn_predictions = bpnn_model.predict(X_test)
    
    baseline_model = SGDRegressor(max_iter=100, random_state=42)
    baseline_model.fit(X_train_full, y_train)
    baseline_predictions = baseline_model.predict(X_test_full)
    
    metrics = {
        "pso_selected_features": selected_features,
        "total_raw_features": len(features_list),
        "hybrid_mae": mean_absolute_error(y_test, bpnn_predictions),
        "hybrid_rmse": np.sqrt(mean_squared_error(y_test, bpnn_predictions)),
        "hybrid_r2": r2_score(y_test, bpnn_predictions),
        "base_mae": mean_absolute_error(y_test, baseline_predictions),
        "base_rmse": np.sqrt(mean_squared_error(y_test, baseline_predictions)),
        "base_r2": r2_score(y_test, baseline_predictions),
        "y_test_observed": y_test,
        "hybrid_predictions": bpnn_predictions,
        "baseline_predictions": baseline_predictions
    }
    
    return metrics