import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="ML Trainer App", 
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: #e6e6fa;
            padding: 1rem;
        }
        .stButton>button {
            background-color: #4b0082;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #6a0dad;
        }
        h1, h2, h3 {
            color: #4b0082;
        }
        .metric-card {
            background-color: #f0f0f0;
            border-radius: 5px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .disclaimer {
            font-size: 0.8em;
            color: #666;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Machine Learning Trainer App")
st.markdown("""
    Easily train and evaluate machine learning models. 
    Ideal for learning, analysis, or deployment.
""", unsafe_allow_html=True)

@st.cache(ttl=3600)
def load_dataset(name):
    return sns.load_dataset(name)

def get_dataset_info(dataset):
    info = {
        "shape": dataset.shape,
        "dtypes": dataset.dtypes,
        "missing": dataset.isnull().sum(),
        "numeric_cols": dataset.select_dtypes(include=["number"]).columns.tolist(),
        "cat_cols": dataset.select_dtypes(exclude=["number"]).columns.tolist()
    }
    return info

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None
if "preprocessing_pipeline" not in st.session_state:
    st.session_state.preprocessing_pipeline = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "model_history" not in st.session_state:
    st.session_state.model_history = []
if "model_type" not in st.session_state:  # Add this line to track model_type in session state
    st.session_state.model_type = None

with st.sidebar.expander("DATASET SELECTION", expanded=True):
    dataset_name = st.selectbox(
        "Choose a dataset", 
        sns.get_dataset_names(),
        help="Select one of the built-in seaborn datasets"
    )
    
    dataset = load_dataset(dataset_name)
    
    uploaded_file = st.file_uploader(
        "Or upload your own CSV", 
        type=["csv"],
        help="Upload a CSV file with your own data"
    )
    
    if uploaded_file:
        try:
            dataset = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded dataset with {dataset.shape[0]} rows and {dataset.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

dataset_info = get_dataset_info(dataset)
st.markdown(f"### Using Dataset: `{dataset_name if not uploaded_file else uploaded_file.name}`")

data_tab1, data_tab2, data_tab3 = st.tabs(["Preview", "Statistics", "Visualization"])

with data_tab1:
    st.dataframe(dataset.head(10), use_container_width=True)
    st.text(f"Shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns")

with data_tab2:
    if st.checkbox("Show Detailed Statistics"):
        st.subheader("Summary Statistics")
        st.write(dataset.describe())
        
        st.subheader("Missing Values")
        missing_data = dataset.isnull().sum()
        if missing_data.sum() > 0:
            st.write(missing_data[missing_data > 0])
        else:
            st.info("No missing values found in the dataset.")
            
        st.subheader("Data Types")
        st.write(pd.DataFrame({'Data Type': dataset.dtypes}))

with data_tab3:
    if len(dataset_info["numeric_cols"]) > 0:
        if len(dataset_info["numeric_cols"]) > 1:
            if st.checkbox("Show Correlation Matrix"):
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = dataset[dataset_info["numeric_cols"]].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(
                    corr_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt=".2f", 
                    cmap="coolwarm", 
                    ax=ax,
                    linewidths=0.5
                )
                st.pyplot(fig)
        
        if st.checkbox("Show Distribution Plots"):
            selected_col = st.selectbox(
                "Select column for distribution", 
                dataset_info["numeric_cols"]
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(dataset[selected_col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

with st.sidebar.expander("FEATURE SELECTION", expanded=True):
    selected_features = st.multiselect(
        "Select features (X)", 
        dataset.columns.tolist(),
        default=dataset_info["numeric_cols"][:3] if len(dataset_info["numeric_cols"]) >= 3 else dataset_info["numeric_cols"],
        help="Select the columns to use as features for your model"
    )
    
    target_feature = st.selectbox(
        "Select target variable (y)", 
        dataset.columns.tolist(),
        index=0 if len(dataset.columns) > 0 else None,
        help="Select the column you want to predict"
    )

with st.sidebar.expander("DATA PREPROCESSING", expanded=True):
    test_size = st.slider(
        "Test set size (%)", 
        10, 50, 20,
        help="Percentage of the dataset to use for testing"
    )
    
    handle_missing = st.selectbox(
        "Handle Missing Values", 
        ["Drop", "Mean/Mode Imputation", "None"],
        help="Choose strategy for handling missing values"
    )
    
    apply_scaling = st.checkbox(
        "Standardize Features",
        help="Scale numerical features to have mean 0 and variance 1"
    )
    
    handle_categorical = st.checkbox(
        "One-Hot Encode Categorical Features",
        help="Convert categorical features to binary indicators"
    )

with st.sidebar.expander("MODEL SELECTION", expanded=True):
    model_option = st.selectbox(
        "Choose Model Type", 
        ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest", "Gradient Boosting"],
        help="Select the type of machine learning model to train"
    )
    
    if model_option in ["Random Forest", "Gradient Boosting"]:
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        max_depth = st.slider("Max Depth", 2, 20, 5, 1)
    
    if model_option in ["Ridge Regression", "Lasso Regression"]:
        alpha = st.slider(
            "Regularization Strength (Alpha)", 
            0.01, 10.0, 1.0, 0.01,
            help="Higher values indicate stronger regularization"
        )
    
    cv_option = st.checkbox(
        "Use Cross-Validation",
        help="Evaluate model on multiple train-test splits"
    )
    
    if cv_option:
        n_folds = st.slider(
            "Number of Folds", 
            2, 10, 5,
            help="Number of cross-validation folds"
        )
    
    use_grid_search = st.checkbox(
        "Use Grid Search for Hyperparameter Tuning",
        help="Automatically search for the best parameters"
    )

can_proceed = selected_features and target_feature and target_feature not in selected_features

if not can_proceed:
    st.warning("Please select features and a target variable (target should not be in features).")
else:
    X = dataset[selected_features].copy()
    y = dataset[target_feature].copy()
    
    is_classification = False
    if target_feature in dataset_info["cat_cols"] or (pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) <= 10):
        is_classification = True
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.info(f"Target classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    numeric_features = [f for f in selected_features if f in dataset_info["numeric_cols"]]
    categorical_features = [f for f in selected_features if f in dataset_info["cat_cols"]]
    
    preprocessors = []
    
    if numeric_features:
        if handle_missing == "Mean/Mode Imputation":
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler() if apply_scaling else 'passthrough')
            ])
        else:
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler() if apply_scaling else 'passthrough')
            ])
        preprocessors.append(('num', numeric_transformer, numeric_features))
    
    if categorical_features and handle_categorical:
        if handle_missing == "Mean/Mode Imputation":
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
        else:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
        preprocessors.append(('cat', categorical_transformer, categorical_features))
    
    if preprocessors:
        preprocessing = ColumnTransformer(transformers=preprocessors)
    else:
        preprocessing = 'passthrough'
    
    if model_option == "Linear Regression":
        model = LinearRegression()
        model_type = "regression"
    elif model_option == "Ridge Regression":
        model = Ridge(alpha=alpha)
        model_type = "regression"
    elif model_option == "Lasso Regression":
        model = Lasso(alpha=alpha)
        model_type = "regression"
    elif model_option == "Random Forest":
        if is_classification:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model_type = "classification"
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model_type = "regression"
    elif model_option == "Gradient Boosting":
        if is_classification:
            model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model_type = "classification"
        else:
            model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model_type = "regression"
    
    # Store model_type in session state
    st.session_state.model_type = model_type
    
    if handle_missing == "Drop":
        combined = pd.concat([X, pd.Series(y, name=target_feature)], axis=1).dropna()
        X = combined[selected_features]
        y = combined[target_feature]
        if len(X) == 0:
            st.error("No data left after dropping missing values. Try using imputation instead.")
            st.stop()
    if len(X) > 0 and len(np.unique(y)) > 1:
        if is_classification:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            num_classes = len(unique_classes)
            test_sample_count = int(len(y) * (test_size / 100))

            # Check if every class has at least 2 samples
            all_classes_valid = np.all(class_counts >= 2)

            # Check if test set can contain at least one of each class
            test_set_large_enough = test_sample_count >= num_classes

            if not all_classes_valid:
                st.warning("Some classes have less than 2 samples — stratified split skipped.")
                stratify_value = None
            elif not test_set_large_enough:
                st.warning(f"Test set too small for stratified split: {test_sample_count} samples < {num_classes} classes.")
                stratify_value = None
            else:
                stratify_value = y
        else:
            stratify_value = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size / 100,
            random_state=42,
            stratify=stratify_value
        )
        st.session_state.feature_columns = X_train.columns.tolist()
        st.session_state.y_test = y_test
        st.session_state.X_test = X_test

        with st.form("model_form"):
            st.write(f"### Configure your {model_option} model")
            st.write(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
            
            if use_grid_search:
                st.write("Grid search will find the best parameters for you.")
                
                if model_option == "Random Forest":
                    param_grid = {
                        'n_estimators': [50, 100, 150],
                        'max_depth': [3, 5, 7, None]
                    }
                elif model_option in ["Ridge Regression", "Lasso Regression"]:
                    param_grid = {
                        'alpha': [0.1, 1.0, 10.0]
                    }
                elif model_option == "Gradient Boosting":
                    param_grid = {
                        'n_estimators': [50, 100, 150],
                        'max_depth': [2, 3, 4],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
                else:
                    param_grid = {}
            
            submitted = st.form_submit_button("Fit Model")
        
        if submitted:
            try:
                with st.spinner("Training model... Please wait."):

                    if preprocessors:
                        preprocessing.fit(X_train)
                        X_train_processed = preprocessing.transform(X_train)
                        X_test_processed = preprocessing.transform(X_test)
                        st.session_state.X_test_processed = X_test_processed 
                        st.session_state.preprocessing_pipeline = preprocessing

                        if hasattr(preprocessing, 'get_feature_names_out'):
                            st.session_state.feature_names = preprocessing.get_feature_names_out()
                        else:
                            st.session_state.feature_names = None
                    else:
                        X_train_processed = X_train
                        X_test_processed = X_test

                    if use_grid_search and param_grid:
                        grid_search = GridSearchCV(
                            model,
                            param_grid,
                            cv=n_folds if cv_option else 3,
                            scoring='neg_mean_squared_error' if st.session_state.model_type == 'regression' else 'accuracy'
                        )
                        grid_search.fit(X_train_processed, y_train)
                        st.session_state.trained_model = grid_search.best_estimator_
                        st.success(f"Best parameters found: {grid_search.best_params_}")

                    elif cv_option:
                        cv_scores = cross_val_score(
                            model, X_train_processed, y_train, cv=n_folds,
                            scoring='neg_mean_squared_error' if st.session_state.model_type == 'regression' else 'accuracy'
                        )
                        st.write(f"Cross-Validation Scores: {cv_scores}")
                        st.write(f"Mean CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

                        model.fit(X_train_processed, y_train)
                        st.session_state.trained_model = model
                    else:
                        model.fit(X_train_processed, y_train)
                        st.session_state.trained_model = model
                        st.session_state.model_history.append({
                        'model_type': model_option,
                        'parameters': str(st.session_state.trained_model.get_params()),
                        'features': selected_features,
                        'target': target_feature
                    })

                    st.success("Model Training Complete!")

            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.info("Try adjusting your feature selection or preprocessing options.")


if st.session_state.trained_model:
    # Load X_test from session
    X_test = st.session_state["X_test"]
    if "feature_columns" in st.session_state:
        for col in st.session_state.feature_columns:
            if col not in X_test.columns:
                X_test[col] = np.nan  # Fill missing columns with NaN
        X_test = X_test[st.session_state.feature_columns]  

if st.session_state.trained_model:
    X_test = st.session_state["X_test"].copy()

    # Get columns the pipeline expects explicitly from ColumnTransformer
    if st.session_state.preprocessing_pipeline and isinstance(st.session_state.preprocessing_pipeline, ColumnTransformer):
        pipeline_columns = []
        for _, _, cols in st.session_state.preprocessing_pipeline.transformers_:
            pipeline_columns.extend(cols)

        # Ensure all expected columns exist in X_test, add missing ones with NaN
        missing_cols = set(pipeline_columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = np.nan

        X_test = X_test[pipeline_columns]
        X_test_processed = st.session_state.preprocessing_pipeline.transform(X_test)

    else:
        X_test_processed = X_test
    y_test = st.session_state.get("y_test", None)

    y_pred = st.session_state.trained_model.predict(X_test_processed)

    # Check for shape mismatch
    if len(y_test) != len(y_pred):
        st.error(f"Prediction length mismatch: y_test ({len(y_test)}), y_pred ({len(y_pred)}).")
        st.stop()

    # Store predictions
    st.session_state.y_pred = y_pred


    result_tab1, result_tab2, result_tab3 = st.tabs(["Performance Metrics", "Visualizations", "Feature Importance"])
    
    with result_tab1:
        st.subheader("Model Performance")
        
        # Use model_type from session state instead of referencing the local variable
        if st.session_state.model_type == "regression":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>R² Score</h3>
                    <h2>{:.3f}</h2>
                </div>
                """.format(r2_score(y_test, y_pred)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>MAE</h3>
                    <h2>{:.3f}</h2>
                </div>
                """.format(mean_absolute_error(y_test, y_pred)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>RMSE</h3>
                    <h2>{:.3f}</h2>
                </div>
                """.format(np.sqrt(mean_squared_error(y_test, y_pred))), unsafe_allow_html=True)
        else:
            col1, col2, col3 = st.columns(3)
            
            if not np.issubdtype(y_pred.dtype, np.integer):
                y_pred = np.round(y_pred).astype(int)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2>{:.3f}</h2>
                </div>
                """.format(accuracy_score(y_test, y_pred)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>Precision</h3>
                    <h2>{:.3f}</h2>
                </div>
                """.format(precision_score(y_test, y_pred, average='weighted')), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>Recall</h3>
                    <h2>{:.3f}</h2>
                </div>
                """.format(recall_score(y_test, y_pred, average='weighted')), unsafe_allow_html=True)
            
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))
            if 'le' in locals() and hasattr(le, 'inverse_transform'):
                st.write("### Class Mapping")
                mapping_df = pd.DataFrame({
                    'Numeric Label': range(len(le.classes_)),
                    'Original Label': le.classes_
                })
                st.dataframe(mapping_df)
    
            if st.checkbox("Show predictions with original labels"):
                try:
                    original_y_test = le.inverse_transform(y_test)
                    original_y_pred = le.inverse_transform(y_pred)
                    st.write(pd.DataFrame({
                        'Original Actual': original_y_test,
                        'Original Predicted': original_y_pred
                    }).head(20))
                except Exception as e:
                    st.error(f"Couldn't convert predictions to original labels: {e}")
    
    with result_tab2:
        st.subheader("Visualizations")
        trained_model = st.session_state.trained_model

        # Use model_type from session state
        if st.session_state.model_type == "regression":
            fig, ax = plt.subplots(figsize=(10, 6))
            residuals = y_test - y_pred
            sns.scatterplot(x=y_pred, y=residuals, ax=ax)
            ax.axhline(y=0, color='r', linestyle='-')
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")
            plt.tight_layout()
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.6)
            
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted Values")
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            
            if len(np.unique(y_test)) == 2 and hasattr(trained_model, "predict_proba"):
                fig, ax = plt.subplots(figsize=(10, 8))
                if 'X_test_processed' not in locals():
                    if st.session_state.preprocessing_pipeline:
                        X_test_processed = st.session_state.preprocessing_pipeline.transform(X_test)
                    else:
                        X_test_processed = X_test
            
                y_pred_prob = trained_model.predict_proba(X_test_processed)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                ax.plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.3f}")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig)
    
    with result_tab3:
        st.subheader("Feature Importance")
        trained_model = st.session_state.trained_model
        if hasattr(trained_model, 'feature_importances_'):
            importances = trained_model.feature_importances_
            
            if st.session_state.feature_names is not None:
                features = st.session_state.feature_names
            else:
                features = selected_features
                
            if len(importances) == len(features):
                feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(15), palette="viridis", ax=ax)
                ax.set_title("Feature Importance")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Feature names don't match importance values. This can happen with one-hot encoding.")
        
        elif hasattr(trained_model, 'coef_'):
            coefs = trained_model.coef_
            
            if len(coefs.shape) > 1 and coefs.shape[0] == 1:
                coefs = coefs[0]
                
            if st.session_state.feature_names is not None:
                features = st.session_state.feature_names
            else:
                features = selected_features
                
            if len(coefs) == len(features):
                coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefs})
                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    x='Coefficient', 
                    y='Feature', 
                    data=coef_df.head(15), 
                    palette="viridis", 
                    ax=ax
                )
                ax.set_title("Feature Coefficients")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Feature names don't match coefficient values.")
        else:
            st.info("Feature importance not available for this model type.")


st.subheader("Export Model")
col1, col2 = st.columns(2)
trained_model = st.session_state.get("trained_model", None)
y_test = st.session_state.get("y_test", None)
y_pred = st.session_state.get("y_pred", None)

with col1:
    model_filename = f"{model_option.lower().replace(' ', '_')}_model.pkl"
    
    try:
        export_data = {
            'model': trained_model,
            'preprocessing': st.session_state.preprocessing_pipeline,
            'feature_names': selected_features,
            'target_name': target_feature,
            'model_type': st.session_state.model_type
        }
        
        joblib.dump(export_data, model_filename)
        
        with open(model_filename, "rb") as file:
            st.download_button(
                "Download Trained Model", 
                data=file, 
                file_name=model_filename,
                help="Download the complete model with preprocessing pipeline"
            )
    except Exception as e:
        st.error(f"Error exporting model: {e}")
        st.info("Try training the model again or check console for details.")
    
    with col2:
        if "y_test" in st.session_state:
            y_test = st.session_state.y_test
            predictions_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            }, index=range(len(y_test)))
            
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                "Download Predictions", 
                data=csv,
                file_name="model_predictions.csv",
                mime="text/csv",
                help="Download actual vs predicted values as CSV"
            )
        else:
            st.warning("No y_test found in session. Please train the model first.")

if st.session_state.model_history and len(st.session_state.model_history) > 0:
    with st.expander("Model Training History"):
        st.write("### Previous Models")
        
        history_df = pd.DataFrame(st.session_state.model_history)
        st.dataframe(history_df)

st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("Developed by Carolina Kogan Plachkinova | Machine Learning Trainer App | Data Visualization and Decision-Making")
