from __future__ import annotations
import pandas as pd
import plotly.express as px

def plot_churn_distribution(df: pd.DataFrame) -> str:
    fig = px.histogram(df, x="Churn", color="Churn", title="Churn Distribution", text_auto=True)
    fig.update_layout(showlegend=False)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def plot_churn_rate_by(df: pd.DataFrame, col: str) -> str:
    tmp = df.groupby(col, dropna=False)["Churn"].mean().reset_index()
    tmp["ChurnRate"] = (tmp["Churn"] * 100).round(2)
    fig = px.bar(tmp, x=col, y="ChurnRate", title=f"Churn Rate by {col}", text="ChurnRate")
    fig.update_traces(texttemplate="%{text}%")
    fig.update_layout(yaxis_title="Churn Rate (%)")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def plot_churn_by_tenure_band(df: pd.DataFrame) -> str:
    tmp = df.copy()
    tmp["tenure_band"] = pd.cut(
        tmp["tenure"],
        bins=[-1, 6, 12, 24, 36, 48, 60, 72],
        labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
    )
    grp = tmp.groupby("tenure_band", dropna=False)["Churn"].mean().reset_index()
    grp["ChurnRate"] = (grp["Churn"] * 100).round(2)
    fig = px.line(grp, x="tenure_band", y="ChurnRate", markers=True, title="Churn Rate by Tenure Band")
    fig.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="Tenure Band (months)")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")