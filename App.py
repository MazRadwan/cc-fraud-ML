import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os.path

# Set up a consistent color palette
fraud_color = '#FFB7B2'    # Light pink for fraud
normal_color = '#A0D2EB'   # Light blue for non-fraud
pastel_colors = ['#A0D2EB', '#FFB7B2', '#D5EEBB']  # For model comparison

def main():
    st.title("Credit Card Fraud Detection Dashboard")
    
    # Load data from notebook results (saved as pickle files)
    data_files = {
        'data': 'notebook_data.pkl',
        'model_results': 'model_results.pkl',
        'feature_importance_rf': 'feature_importance_rf.pkl',
        'feature_importance_xgb': 'feature_importance_xgb.pkl',
    }
    
    # Check if files exist
    if all(os.path.exists(f) for f in data_files.values()):
        # Load all data
        df_clean = pd.read_pickle(data_files['data'])
        model_results = pd.read_pickle(data_files['model_results'])
        feature_importance_rf = pd.read_pickle(data_files['feature_importance_rf'])
        feature_importance_xgb = pd.read_pickle(data_files['feature_importance_xgb'])
        
        # Display tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "Data Exploration", "Feature Importance", "PCA Analysis"])
        
        with tab1:
            st.header("Model Performance Comparison")
            show_model_comparison(model_results)
            
        with tab2:
            st.header("Transaction Data Analysis")
            show_kde_violin_plot(df_clean)
            show_box_plot(df_clean)
            show_correlation_heatmap(df_clean)
            
        with tab3:
            st.header("Feature Importance Analysis")
            show_feature_importance(feature_importance_rf, "Random Forest", pastel_colors[0])
            show_feature_importance(feature_importance_xgb, "XGBoost", pastel_colors[1])
            
        with tab4:
            st.header("PCA Features Analysis")
            show_pca_histograms(df_clean)
    else:
        st.error("Required data files not found! Please run the notebook first to generate result files.")
        st.info("The dashboard needs the following files from your notebook:")
        for key, filename in data_files.items():
            st.code(f"df_clean.to_pickle('{filename}')  # Save your processed data")
        st.info("Add these lines to your notebook and run it before launching the dashboard.")

# New function to display model comparisons
def show_model_comparison(model_results):
    """Display model performance metrics comparison."""
    # AUC-ROC Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = sns.barplot(
        x='Mean AUC-ROC', 
        y='Model', 
        data=model_results,
        palette=pastel_colors,
        ax=ax
    )
    
    # Add data labels
    for p in bars.patches:
        width = p.get_width()
        ax.text(width + 0.0001, 
                p.get_y() + p.get_height()/2, 
                f'{width:.6f}',
                ha='left', 
                va='center', 
                fontweight='bold', 
                size=10)
    
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Mean AUC-ROC', fontsize=14)
    ax.set_ylabel('Model', fontsize=14)
    ax.set_xlim(0.9997, 1.0001)  # Adjusted to focus on differences
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
    
    # Training Time Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(
        x='Training Time (sec)', 
        y='Model', 
        data=model_results,
        palette=pastel_colors,
        ax=ax
    )
    
    # Add data labels
    for p in bars.patches:
        width = p.get_width()
        ax.text(width + 1, 
                p.get_y() + p.get_height()/2, 
                f'{width:.1f}s',
                ha='left', 
                va='center', 
                fontweight='bold', 
                size=10)
    
    ax.set_title('Model Training Time Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Time (seconds)', fontsize=14)
    ax.set_ylabel('Model', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
    
    # Display raw numbers in a table
    st.subheader("Model Performance Metrics")
    st.dataframe(model_results.style.highlight_max(axis=0, subset=['Mean AUC-ROC']).highlight_min(axis=0, subset=['Training Time (sec)']), use_container_width=True)

# Function for feature importance visualizations
def show_feature_importance(feature_importance, model_name, color):
    """Display feature importance visualization."""
    st.subheader(f"{model_name} Feature Importance")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot top 15 features
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=feature_importance.head(15), 
        color=color,
        ax=ax
    )
    ax.set_title(f'{model_name} Feature Importance', fontsize=16)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)

# Your existing visualization functions
def show_kde_violin_plot(df_clean):
    """Recreates the KDE and Violin plots for the Amount feature."""
    st.subheader("Transaction Amount Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: KDE Plot ---
    sns.kdeplot(
        data=df_clean[df_clean.Class == 0], x='Amount',
        fill=True, color=normal_color, label='Normal (0)',
        log_scale=True, alpha=0.3, ax=axes[0]
    )
    sns.kdeplot(
        data=df_clean[df_clean.Class == 1], x='Amount',
        fill=True, color=fraud_color, label='Fraud (1)',
        log_scale=True, alpha=0.3, ax=axes[0]
    )
    axes[0].set_title('KDE of Transaction Amount (log scale)', fontsize=14)
    axes[0].set_xlabel('Amount', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].legend()

    # --- Subplot 2: Violin Plot ---
    sns.violinplot(
        x='Class',
        y='Amount',
        data=df_clean,
        palette=[normal_color, fraud_color],
        scale='width',
        ax=axes[1]
    )
    axes[1].set_title('Amount by Class (log scale)', fontsize=14)
    axes[1].set_xlabel('Class (0: Normal, 1: Fraud)', fontsize=12)
    axes[1].set_ylabel('Amount', fontsize=12)
    axes[1].set_yscale('log')

    plt.tight_layout()
    st.pyplot(fig)

# Keep your other visualization functions unchanged...
def show_box_plot(df_clean):
    """Displays a box plot of Amount by Class."""
    st.subheader("Box Plot of Transaction Amount by Class")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.boxplot(
        x='Class',
        y='Amount',
        data=df_clean,
        palette=[normal_color, fraud_color],
        ax=ax
    )
    ax.set_title('Transaction Amount by Class', fontsize=14)
    ax.set_xlabel('Class (0: Normal, 1: Fraud)', fontsize=12)
    ax.set_ylabel('Amount', fontsize=12)
    ax.set_yscale('log')

    plt.tight_layout()
    st.pyplot(fig)

def show_pca_histograms(df_clean):
    """Plots histograms for selected PCA features by class."""
    st.subheader("Histograms of Selected PCA Features")
    selected_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 
                         'V7', 'V8', 'V9', 'V10', 'V11', 'V12']
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(selected_features):
        sns.histplot(
            data=df_clean,
            x=feature,
            hue='Class',
            kde=True,
            bins=30,
            palette=[normal_color, fraud_color],
            alpha=0.6,
            ax=axes[i]
        )
        axes[i].set_title(f'Distribution of {feature}', fontsize=12)
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)
        # Hide per-subplot legends to reduce clutter
        axes[i].legend([], frameon=False)

    plt.suptitle('Distributions of First 12 PCA Features by Class', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    st.pyplot(fig)

def show_correlation_heatmap(df_clean):
    """Displays a correlation heatmap of the dataset."""
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 12))
    
    correlation_matrix = df_clean.corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(
        correlation_matrix, 
        mask=mask, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0, 
        square=True, 
        linewidths=.5, 
        annot=False, 
        fmt='.2f', 
        cbar_kws={'shrink': .7},
        ax=ax
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=16)
    
    plt.tight_layout()
    st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()