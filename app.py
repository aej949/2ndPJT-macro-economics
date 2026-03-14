import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import koreanize_matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="거시 지표 & 업종 ETF 상관관계 분석", layout="wide")

st.title("거시 지표(금리·원자재)와 업종별 성장 상관관계 분석 대시보드")
st.markdown("""
**📝 프로젝트 목적**
기준금리 및 3대 핵심 원자재(금, 은, 원유)의 가격 변동이 주식 시장 내 특정 업종(ETF)의 주가 흐름에 미치는 영향을 객관적 데이터로 검증하고 시각화합니다.
*(본 대시보드는 팩트 기반 시각화를 유지하며, 상관된 요인이 무조건 수익률로 보장된다는 낙관적 시나리오를 배제합니다.)*
""")

# --- 1. Data Fetching & Preprocessing ---
@st.cache_data(ttl=3600)
def load_and_preprocess_data(start_date, end_date):
    # 1. Macro Indicators (FRED via pandas_datareader)
    # DGS10 (US 10-Year Treasury Yield), FEDFUNDS (Fed Funds Rate) proxy
    fred_series = ['DGS10', 'FEDFUNDS']
    try:
        df_macro = web.DataReader(fred_series, 'fred', start_date, end_date)
        df_macro.columns = ['미국 10년물 국채(%)', '미국 기준금리(%)']
        df_macro.ffill(inplace=True)
    except Exception as e:
        df_macro = pd.DataFrame(columns=['미국 10년물 국채(%)', '미국 기준금리(%)'], index=pd.date_range(start_date, end_date))

    # 2. Commodities (Gold, Silver, WTI) via yfinance
    tickers_comm = {'GC=F': '금(Gold)', 'SI=F': '은(Silver)', 'CL=F': '유가(WTI)'}
    df_comm = pd.DataFrame()
    for tkr, name in tickers_comm.items():
        data = yf.download(tkr, start=start_date, end=end_date)['Close']
        if isinstance(data, pd.DataFrame):
            df_comm[name] = data.iloc[:, 0]
        else:
            df_comm[name] = data
    
    # 3. 주요 업종별 ETF (KRX 기반 대리변수 활용, yfinance 사용)
    # 금융 139270.KS, 건설 139280.KS, IT 139260.KS, 에너지 139250.KS
    etf_map = {
        '139270.KS': '금융(Finance) ETF',
        '139280.KS': '건설(Construction) ETF',
        '139260.KS': 'IT(KOSPI) ETF',
        '139250.KS': '에너지/화학(Energy) ETF'
    }
    df_etf = pd.DataFrame()
    for tk, name in etf_map.items():
        try:
            etf_data = yf.download(tk, start=start_date, end=end_date)['Close']
            if isinstance(etf_data, pd.DataFrame):
                df_etf[name] = etf_data.iloc[:, 0]
            else:
                df_etf[name] = etf_data
        except Exception:
            df_etf[name] = np.nan

    # 4. Merge Data (Business Day Index 기준 시계열 동기화)
    b_days = pd.date_range(start=start_date, end=end_date, freq='B')
    df_merged = pd.DataFrame(index=b_days)
    df_merged = df_merged.join(df_macro, how='left')
    df_merged = df_merged.join(df_comm, how='left')
    df_merged = df_merged.join(df_etf, how='left')

    # Forward fill (결측치는 직전 영업일 데이터로 채움)
    df_merged = df_merged.ffill()
    df_merged = df_merged.bfill()  # 맨 앞 비어있는 경우 대상
    df_merged.index.name = "날짜 (Business Day)"
    return df_merged

# --- Sidebar Controls ---
st.sidebar.header("조회 조건 설정")
today = datetime.today()
default_start = today - relativedelta(years=3)
start_date_input = st.sidebar.date_input("조회 시작일", default_start)
end_date_input = st.sidebar.date_input("조회 종료일", today)

start_date = start_date_input.strftime("%Y-%m-%d")
end_date = end_date_input.strftime("%Y-%m-%d")

if start_date_input >= end_date_input:
    st.sidebar.error("시작일은 종료일보다 이전이어야 합니다.")
    st.stop()

with st.spinner('데이터를 수집 및 전처리 중입니다... (최대 3초 소요)'):
    df_raw = load_and_preprocess_data(start_date, end_date)

# Base=100 지수화 적용 (금리는 단위가 % 이므로 제외)
df_indexed = df_raw.copy()
cols_to_index = ['금(Gold)', '은(Silver)', '유가(WTI)', '금융(Finance) ETF', '건설(Construction) ETF', 'IT(KOSPI) ETF', '에너지/화학(Energy) ETF']

for col in cols_to_index:
    base_val = df_indexed[col].dropna().iloc[0] if not df_indexed[col].dropna().empty else 1
    if base_val != 0:
        df_indexed[col] = (df_indexed[col] / base_val) * 100

st.markdown("---")

# ================================
# 패널 A: 거시 지표 & 원자재 트렌드 보드
# ================================
st.header("📌 패널 A: 거시 지표 & 원자재 트렌드 보드")
st.markdown("거시 지표(우측 Y축, %)와 주요 원자재 지수(좌측 Y축, Base 100)의 복합 시계열입니다.")

macro_cols = ['미국 10년물 국채(%)', '미국 기준금리(%)']
comm_cols = ['금(Gold)', '은(Silver)', '유가(WTI)']

selected_macros = st.multiselect("거시 지표 (금리 선택)", macro_cols, default=['미국 10년물 국채(%)'])
selected_comms = st.multiselect("원자재 지수", comm_cols, default=['금(Gold)', '유가(WTI)'])

fig_a = make_subplots(specs=[[{"secondary_y": True}]])

for col in selected_comms:
    fig_a.add_trace(
        go.Scatter(x=df_indexed.index, y=df_indexed[col], name=col, mode='lines'),
        secondary_y=False
    )

for col in selected_macros:
    fig_a.add_trace(
        go.Scatter(x=df_indexed.index, y=df_indexed[col], name=col, mode='lines', line=dict(dash='dot')),
        secondary_y=True
    )

fig_a.update_layout(
    title="원자재 지수 트렌드 및 기준 금리 변동 (거시 환경)",
    xaxis=dict(title="영업일 기준 일자"),
    yaxis=dict(title="원자재 지수 (Base=100)"),
    yaxis2=dict(title="금리 (%)", overlaying="y", side="right"),
    height=450,
    hovermode="x unified"
)
st.plotly_chart(fig_a, use_container_width=True)

st.markdown("---")

# ================================
# 패널 B: 업종별 상관관계 매트릭스
# ================================
st.header("📌 패널 B: 업종별 상관관계 매트릭스 (Pearson Correlation)")

etf_cols = ['금융(Finance) ETF', '건설(Construction) ETF', 'IT(KOSPI) ETF', '에너지/화학(Energy) ETF']
indep_vars = macro_cols + comm_cols

if not df_raw[etf_cols].dropna(how='all').empty:
    corr_matrix = df_raw[indep_vars + etf_cols].corr()
    filtered_corr = corr_matrix.loc[indep_vars, etf_cols]

    fig_b = px.imshow(
        filtered_corr, 
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="거시 환경 변수 - ETF 종속 변수 간의 피어슨 상관계수 히트맵"
    )
    fig_b.update_layout(height=450)
    st.plotly_chart(fig_b, use_container_width=True)
    st.info("💡 **해석 방법**: 1에 가까울수록 진한 붉은색(동행), -1에 가까울수록 진한 푸른색(역행)을 의미합니다. 설정된 조회 기간 내에서 어떤 업종이 금리나 유가 등과 가장 많이, 반대로 혹은 함께 움직였는지 파악하세요.")
else:
    st.warning("선택된 기간의 ETF 데이터를 불러올 수 없어 상관계수 렌더링을 생략합니다.")

st.markdown("---")

# ================================
# 패널 C: 특정 업종 상세 분석
# ================================
st.header("📌 패널 C: 특정 업종의 핵심 상관 지표 Drill-Down 및 실적 트렌드")

selected_etf = st.selectbox("분석 대상 업종 ETF 선택", etf_cols)

if not df_raw[etf_cols].dropna(how='all').empty:
    corr_series = filtered_corr[selected_etf].abs().sort_values(ascending=False)
    top_2_indicators = corr_series.index[:2].tolist()

    st.subheader(f"{selected_etf} & 동조율 최상위 지표 2개 (Base=100 비교)")
    st.markdown(f"해당 ETF와 가장 상관성이 높았던 지표는 **{top_2_indicators[0]}**, **{top_2_indicators[1]}** 입니다.")

    fig_c = make_subplots(specs=[[{"secondary_y": True}]])
    
    # ETF trace (Base 100 indicator)
    fig_c.add_trace(
        go.Scatter(x=df_indexed.index, y=df_indexed[selected_etf], name=selected_etf, line=dict(color='black', width=3)),
        secondary_y=False
    )
    
    colors = ['firebrick', 'royalblue']
    for i, indicator in enumerate(top_2_indicators):
        is_macro = indicator in macro_cols
        # If macro, we still plot original raw rate %; if indicator is comm we use the base100 index
        plot_y = df_raw[indicator] if is_macro else df_indexed[indicator]
        
        fig_c.add_trace(
            go.Scatter(x=df_indexed.index, y=plot_y, name=indicator, line=dict(color=colors[i], dash='dash')),
            secondary_y=is_macro
        )

    fig_c.update_layout(
        title=f"{selected_etf} 기간별 움직임과 최상위 상관지표 간 동행성 검증",
        yaxis_title="지수 변동 (Base=100)",
        yaxis2_title="금리(%)" if any(x in macro_cols for x in top_2_indicators) else "",
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig_c, use_container_width=True)

    # 하단 DART Mock 데이터 표시 (분기별 영업이익률 증감 추정)
    st.markdown("**📊 해당 업종 대표 기업 분기별 영업이익률 증감 추정치 (Mock Data)**")
    q_trend_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    
    # 팩트 기반 시각화를 유지하기 위해 난수로 괴리율을 표현할 때에도 건조함을 유지
    np.random.seed(hash(selected_etf) % 2**32) 
    mock_profits = np.random.uniform(low=-3.0, high=8.0, size=len(q_trend_dates))

    fig_bar = px.bar(
        x=q_trend_dates, 
        y=mock_profits, 
        labels={'x': '분기', 'y': '영업이익률 증감 트렌드 (%)'},
        text_auto=".1f"
    )
    fig_bar.update_traces(marker_color=['#2ca02c' if x >= 0 else '#d62728' for x in mock_profits])
    fig_bar.update_layout(height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("""
> ⚠️ **할루시네이션(환각) 방지 및 주의 안내**
> - 위 시각화는 팩트(Fact) 데이터 간의 상관성을 계산한 객관적 지표에 불과하며, 실적 펀더멘털의 직접적인 보장을 의미하지 않습니다.
> - 상관계수는 통계적 산출값으로 유가가 오르더라도 에너지 ETF가 하락하는 '관계 단절' 구간이 존재할 수 있습니다. 
> - 절대 특정 ETF에 대한 투자의견(Buy/Sell)이나 장래 예측치를 내포하지 않습니다.
""")
