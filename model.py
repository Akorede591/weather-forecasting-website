import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================================
# PHASE 1: BINARY PARTICLE SWARM OPTIMIZATION (BPSO) FEATURE SELECTION
# =========================================================================
class BPSOFeatureSelection:
    def __init__(self, X, y, num_particles=10, max_iter=10, alpha=0.85):
        """
        Initializes the Binary PSO optimization engine.
        X: Scaled input features matrix (numpy array)
        y: Target forecast variable vector (numpy array)
        alpha: Weight balancing metric accuracy vs feature reduction space
        """
        self.X = X
        self.y = y
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.alpha = alpha
        self.num_features = X.shape[1]
        
        # Internal validation split to calculate the fitness score of each particle
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
    def _sigmoid(self, x):
        """Maps continuous velocity values to a 0-1 probability space."""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def _calculate_fitness(self, position):
        """Evaluates how well a specific subset of features performs."""
        selected_indices = np.where(position == 1)[0]
        
        # Penalty block: If particle selects zero features, return maximum cost
        if len(selected_indices) == 0:
            return float('inf')
        
        # Extract the subset feature matrix for this particle
        X_tr_sub = self.X_train[:, selected_indices]
        X_val_sub = self.X_val[:, selected_indices]
        
        # High-speed estimator used to rapidly calculate prediction cost
        evaluator = SGDRegressor(max_iter=20, random_state=42, tol=1e-3)
        try:
            evaluator.fit(X_tr_sub, self.y_train)
            mse = mean_squared_error(self.y_val, evaluator.predict(X_val_sub))
        except:
            mse = float('inf')
            
        # Fitness Function Equation: alpha * MSE + (1 - alpha) * Feature_Ratio
        feature_ratio = len(selected_indices) / self.num_features
        fitness_score = (self.alpha * mse) + ((1 - self.alpha) * feature_ratio)
        return fitness_score

    def optimize(self, progress_callback=None):
        """Executes the evolutionary swarm search to find the optimal feature mask."""
        # Initialize continuous velocity arrays between bounds [-4, 4]
        velocities = np.random.uniform(-4, 4, (self.num_particles, self.num_features))
        # Initialize binary positions (0 or 1) randomly
        positions = np.random.randint(2, size=(self.num_particles, self.num_features))
        
        # Tracking vectors for Personal Bests (p_best) and Global Best (g_best)
        p_best_positions = np.copy(positions)
        p_best_scores = np.array([self._calculate_fitness(p) for p in positions])
        
        g_best_idx = np.argmin(p_best_scores)
        g_best_position = np.copy(p_best_positions[g_best_idx])
        g_best_score = p_best_scores[g_best_idx]
        
        # Classic PSO Parameters
        w, c1, c2 = 0.7, 1.5, 1.5
        
        for iteration in range(self.max_iter):
            r1 = np.random.rand(self.num_particles, self.num_features)
            r2 = np.random.rand(self.num_particles, self.num_features)
            
            # 1. Update Velocity Vector
            cognitive = c1 * r1 * (p_best_positions - positions)
            social = c2 * r2 * (g_best_position - positions)
            velocities = w * velocities + cognitive + social
            velocities = np.clip(velocities, -4, 4) # Guard against velocity explosion
            
            # 2. Map Velocity to Probability and Update Position Vector
            probabilities = self._sigmoid(velocities)
            positions = (np.random.rand(self.num_particles, self.num_features) < probabilities).astype(int)
            
            # 3. Update Personal Best matrices
            for i in range(self.num_particles):
                current_score = self._calculate_fitness(positions[i])
                if current_score < p_best_scores[i]:
                    p_best_scores[i] = current_score
                    p_best_positions[i] = np.copy(positions[i])
                    
            # 4. Update Global Best matrix
            best_iter_idx = np.argmin(p_best_scores)
            if p_best_scores[best_iter_idx] < g_best_score:
                g_best_score = p_best_scores[best_iter_idx]
                g_best_position = np.copy(p_best_positions[best_iter_idx])
            
            # Optional UI progress updater handle
            if progress_callback:
                progress_callback(iteration + 1, self.max_iter, g_best_score)
            
        return g_best_position


# =========================================================================
# PHASE 2: PIPELINE EXECUTION ENGINE (BPSO + BPNN TRAINING LOOP)
# =========================================================================
def run_hybrid_pipeline(df, target_column, excluded_cols, swarm_size, pso_iters, bpnn_topology):
    """
    Executes the entire workflow pipeline:
    Data Preprocessing -> BPSO Feature Selection -> BPNN Model Core Training -> Evaluation
    """
    # Clean non-numeric tracking metadata columns
    working_df = df.drop(columns=[col for col in excluded_cols if col in df.columns], errors='ignore')
    clean_df = working_df.dropna()
    
    # Isolate targets and features lists
    features_list = [col for col in clean_df.columns if col != target_column]
    X_raw = clean_df[features_list].select_dtypes(include=[np.number]).values
    y_raw = clean_df[target_column].values
    
    if X_raw.shape[1] == 0:
        raise ValueError("Dataset does not contain valid numeric attributes to process.")
        
    # Scale feature dimensions to uniform range [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Execute Phase 1: BPSO Optimization
    pso_engine = BPSOFeatureSelection(X_scaled, y_raw, num_particles=swarm_size, max_iter=pso_iters)
    optimal_mask = pso_engine.optimize()
    selected_features = [features_list[i] for i, mask in enumerate(optimal_mask) if mask == 1]
    
    if len(selected_features) == 0:
        raise RuntimeError("BPSO optimization failed to identify any relevant feature descriptors.")
        
    # Extract the optimized mathematical feature subset space
    X_optimized = X_scaled[:, np.where(optimal_mask == 1)[0]]
    
    # Train-Test Validation Matrix Splits (80% Training, 20% Test Evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X_optimized, y_raw, test_size=0.2, random_state=42)
    X_train_full, X_test_full, _, _ = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42)
    
    # Execute Phase 2: Train Backpropagation Neural Network Regressor
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
    
    # Train Standard Baseline Model for Performance Invariant Comparison
    baseline_model = SGDRegressor(max_iter=100, random_state=42)
    baseline_model.fit(X_train_full, y_train)
    baseline_predictions = baseline_model.predict(X_test_full)
    
    # Calculate Statistical Core Performance Analytics Metas
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