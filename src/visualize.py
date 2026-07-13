import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_churn_distribution(df):
    plt.figure(figsize=(6, 4))
    df['Churn'].value_counts().plot(kind='bar')
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Not Churned', 'Churned'], rotation=0)
    plt.savefig('churn_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(20, 16))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(precision, recall, threshold):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.axvline(x=threshold, color='r', linestyle='--',
                label=f'Optimal threshold={threshold}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
    plt.show()