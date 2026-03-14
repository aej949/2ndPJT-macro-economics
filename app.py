import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="거시 지표 & 업종 ETF 상관관계 분석 v1.1", layout="wide")

st.title("거시 지표(금리·원자재)와 업종별 성장 상관관계 분석 대시보드 v1.1")
st.markdown("""
**📝 프로젝트 목적 및 개선 배경**
기준금리 및 3대 핵심 원자재의 변동이 주식 시장 내 핵심 업종 ETF에 미치는 영향을 데이터 기반으로 검증합니다.
*v1.1 업데이트:* 단순 가격 절대치 비교가 아닌 **'변동률(Return %)' 기반의 상관관계**를 계산하고, 거시 경제가 시장에 미치는 **시차(Time-Lag)**를 반영할 수 있도록 분석 로직을 고도화하였습니다.
""")

# --- 1. Data Fetching & Preprocessing ---
@st.cache_data(ttl=3600)
def load_and_preprocess_data(start_date, end_date):
    # 1. Macro Indicators (FRED via Pandas CSV HTTP request)
    try:
        dgs10 = pd.read_csv(
            'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10',
            index_col=0, parse_dates=True, na_values='.'
        )
        fedfunds = pd.read_csv(
            'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS',
            index_col=0, parse_dates=True, na_values='.'
        )
        df_macro = dgs10.join(fedfunds, how='outer')
        df_macro = df_macro.loc[start_date:end_date]
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
    
    # 3. 고도화된 주요 업종별 ETF 
    etf_map = {
        '091220.KS': 'KODEX 은행(금리 수혜)',
        '091230.KS': 'KODEX 건설(금리 피해)',
        '091250.KS': 'KODEX 에너지화학(원유)',
        '091240.KS': 'KODEX 철강(산업금속)',
        '305540.KS': 'TIGER 2차전지테마(성장)',
        '091160.KS': 'KODEX 반도체(유동성)',
        '204420.KS': 'KODEX 배당성장(방어)'
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

    df_merged = df_merged.ffill().bfill()
    df_merged.index.name = "날짜 (Business Day)"
    return df_merged

# --- Sidebar Controls ---
st.sidebar.header("조회 및 분석 조건 설정")
today = datetime.today()
default_start = today - relativedelta(years=5)  # YoY 계산 등을 위해 기본 기간 확장
start_date_input = st.sidebar.date_input("조회 시작일", default_start)
end_date_input = st.sidebar.date_input("조회 종료일", today)

start_date = start_date_input.strftime("%Y-%m-%d")
end_date = end_date_input.strftime("%Y-%m-%d")

if start_date_input >= end_date_input:
    st.sidebar.error("시작일은 종료일보다 이전이어야 합니다.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("분석 로직 설정 (요건 B & C)")
return_type = st.sidebar.radio(
    "변동률 기준 선택 (상관관계 계산용)",
    ("월간 변동률 (1 Month Return)", "전년 동기 대비 변동률 (YoY)")
)
lag_selection = st.sidebar.selectbox(
    "거시 지표 시차(Time-Lag) 적용",
    ("0개월 (동행)", "3개월 후행", "6개월 후행")
)

# Parse parameters
lag_months = int(lag_selection.split("개월")[0])
lag_days = lag_months * 21  # 1달을 대략 21 영업일로 산정
pct_periods = 21 if "월간" in return_type else 252  # 1달=21일, 1년=252일

with st.spinner('데이터 수집 및 변동률/시차 변환 중입니다...'):
    df_raw = load_and_preprocess_data(start_date, end_date)

# Base=100 지수화 적용 (패널 A 시각화용)
df_indexed = df_raw.copy()
macro_cols = ['미국 10년물 국채(%)', '미국 기준금리(%)']
comm_cols = ['금(Gold)', '은(Silver)', '유가(WTI)']
etf_cols = [
    'KODEX 은행(금리 수혜)', 'KODEX 건설(금리 피해)', 'KODEX 에너지화학(원유)', 
    'KODEX 철강(산업금속)', 'TIGER 2차전지테마(성장)', 'KODEX 반도체(유동성)', 'KODEX 배당성장(방어)'
]

cols_to_index = comm_cols + etf_cols
for col in cols_to_index:
    base_val = df_indexed[col].dropna().iloc[0] if not df_indexed[col].dropna().empty else 1
    if base_val != 0:
        df_indexed[col] = (df_indexed[col] / base_val) * 100

st.markdown("---")

# ================================
# 패널 A: 거시 지표 & 원자재 트렌드 보드
# ================================
st.header("📌 패널 A: 거시 지표 & 원자재 시계열 트렌드")
st.markdown("거시 지표 절대 수치(우측 Y축, %)와 주요 원자재 가격 지수(좌측 Y축, Base 100)의 복합 시계열입니다.")

selected_macros = st.multiselect("거시 지표 (금리 선택)", macro_cols, default=['미국 10년물 국채(%)'])
selected_comms = st.multiselect("원자재 지수", comm_cols, default=['금(Gold)', '유가(WTI)'])

fig_a = make_subplots(specs=[[{"secondary_y": True}]])
for col in selected_comms:
    fig_a.add_trace(go.Scatter(x=df_indexed.index, y=df_indexed[col], name=col, mode='lines'), secondary_y=False)
for col in selected_macros:
    fig_a.add_trace(go.Scatter(x=df_indexed.index, y=df_indexed[col], name=col, mode='lines', line=dict(dash='dot')), secondary_y=True)

fig_a.update_layout(
    title="원자재 지수 트렌드 및 기준 금리 변동",
    xaxis_title="영업일 기준 일자", yaxis_title="원자재 지수 (Base=100)", yaxis2_title="금리 (%)",
    height=450, hovermode="x unified"
)
st.plotly_chart(fig_a, use_container_width=True)
st.markdown("---")

# ================================
# 변동률 및 시차 데이터 변환 로직 (Data Transformation)
# ================================
df_transformed = df_raw.copy()
indep_vars = macro_cols + comm_cols
for col in indep_vars:
    if col in macro_cols:
        df_transformed[col] = df_transformed[col].diff(periods=pct_periods) # 금리는 %p 변동
    else:
        df_transformed[col] = df_transformed[col].pct_change(periods=pct_periods) * 100

for col in etf_cols:
    df_transformed[col] = df_transformed[col].pct_change(periods=pct_periods) * 100

# 2. 시차(Lag) 적용: 독립변수(거시지표/원자재)를 미래로 밀어서 후행하는 종속변수(ETF)와 시점을 맞춤
if lag_days > 0:
    for col in indep_vars:
        df_transformed[col] = df_transformed[col].shift(lag_days)

df_transformed.dropna(inplace=True)

# ================================
# 패널 B: 업종별 상관관계 매트릭스
# ================================
st.header("📌 패널 B: 변동률 & 시차 기반 핵심 섹터 상관관계 매트릭스")
st.markdown(f"**활성화된 필터:** 기준={return_type} / 시차={lag_selection}")

if not df_transformed[etf_cols].empty:
    corr_matrix = df_transformed[indep_vars + etf_cols].corr()
    filtered_corr = corr_matrix.loc[indep_vars, etf_cols]

    fig_b = px.imshow(
        filtered_corr, 
        text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title=f"거시 지표와 ETF 섹터 간의 피어슨 상관계수 (Lag: {lag_selection})"
    )
    fig_b.update_layout(height=500)
    st.plotly_chart(fig_b, use_container_width=True)
    st.info("💡 **해석 방법**: 단순 가격 비교가 아닌 '변동률(수익률)'의 동행성을 측정합니다. 1에 가까울수록 진한 붉은색(양의 상관), -1에 가까울수록 진한 푸른색(음의 상관)입니다.")
else:
    st.warning("선택된 기간/필터 조합으로 데이터를 충분히 산출할 수 없어 상관계수 렌더링을 생략합니다. 조회 시작일을 더 과거로 설정해보세요.")
st.markdown("---")

# ================================
# 패널 C: 특정 업종 상세 변동률 분석
# ================================
st.header("📌 패널 C: 섹터별 상세 변동률 Drill-Down")

selected_etf = st.selectbox("분석 대상 섹터 ETF 선택", etf_cols)

if not df_transformed.empty:
    corr_series = filtered_corr[selected_etf].abs().sort_values(ascending=False)
    top_2_indicators = corr_series.index[:2].tolist()

    st.subheader(f"{selected_etf} & 동조율 최상위 지표 변동 트렌드")
    st.markdown(f"해당 ETF 변동과 가장 상관계수가 높은 지표(시차 적용)는 **{top_2_indicators[0]}**, **{top_2_indicators[1]}** 입니다.")

    fig_c = make_subplots(specs=[[{"secondary_y": True}]])
    
    # ETF trace (변동률)
    fig_c.add_trace(go.Scatter(x=df_transformed.index, y=df_transformed[selected_etf], name=f"{selected_etf} 변동률(%)", line=dict(color='black', width=3)), secondary_y=False)
    
    colors = ['firebrick', 'royalblue']
    for i, indicator in enumerate(top_2_indicators):
        is_macro = indicator in macro_cols
        # plot variable
        plot_y = df_transformed[indicator]
        
        fig_c.add_trace(go.Scatter(x=df_transformed.index, y=plot_y, name=f"{indicator} (Lag={lag_months}M)", line=dict(color=colors[i], dash='dash')), secondary_y=is_macro)

    fig_c.update_layout(
        title=f"{selected_etf} 변동률과 최상위 상관지표 변동 간 동행성 검증",
        yaxis_title="ETF 및 원자재 변동률 (%)", yaxis2_title="금리 변동 (%p)" if any(x in macro_cols for x in top_2_indicators) else "",
        height=450, hovermode="x unified"
    )
    st.plotly_chart(fig_c, use_container_width=True)

    # 하단 DART Mock 데이터 표시 (분기별 영업이익률 증감 추정)
    st.markdown("**📊 해당 섹터 대표 기업 분기별 영업이익률 증감 추정치 (Mock Data)**")
    q_trend_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    
    np.random.seed(hash(selected_etf) % 2**32) 
    mock_profits = np.random.uniform(low=-5.0, high=12.0, size=len(q_trend_dates))

    fig_bar = px.bar(
        x=q_trend_dates, y=mock_profits, labels={'x': '분기', 'y': '영업이익 변동률 추정 (%)'}, text_auto=".1f"
    )
    fig_bar.update_traces(marker_color=['#2ca02c' if x >= 0 else '#d62728' for x in mock_profits])
    fig_bar.update_layout(height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("""
> ⚠️ **분석의 한계점 및 주의사항**
> - (데이터 제약) TIGER 2차전지테마(305540) 등 상대적으로 최근에 상장된 ETF는 설정 이전의 과거 데이터(예: 5년 전)가 결측치로 처리되어 해당 기간의 상관관계 계산 모수가 적어질 수 있습니다.
> - (변동률 왜곡) 금리 등 거시지표의 변동성 폭에 비해 개별 ETF의 주가 변동성이 크므로 차트의 Y축 스케일을 유의하여 해석해야 합니다.
> - 상관성은 인과관계를 의미하지 않으며, 투자 권유를 뜻하지 않습니다.
""")
