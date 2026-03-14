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
    
    # 3. 고도화된 주요 업종별 ETF (극단적 역/정 상관관계 보유)
    etf_map = {
        'XLE': '에너지(XLE, 원유+)',
        'JETS': '항공(JETS, 원유-)',
        'GDX': '금광업(GDX, 금+)',
        'XLF': '금융(XLF, 금리+)',
        'VNQ': '리츠/부동산(VNQ, 금리-)'
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
    '에너지(XLE, 원유+)', '항공(JETS, 원유-)', '금광업(GDX, 금+)', 
    '금융(XLF, 금리+)', '리츠/부동산(VNQ, 금리-)'
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

# 모두 NaN인 컬럼(에러로 수집되지 않은 ETF 등)을 먼저 버린 후 행 결측치 제거
df_transformed.dropna(axis=1, how='all', inplace=True)
df_transformed.dropna(inplace=True)

# 파싱 성공한 목록으로 컬럼 목록 갱신
etf_cols = [c for c in etf_cols if c in df_transformed.columns]
indep_vars = [c for c in indep_vars if c in df_transformed.columns]

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
    
    # 수익률 격차(Spread) Area Chart 추가 (1순위 지표 대상)
    prime_indicator = top_2_indicators[0]
    is_macro = prime_indicator in macro_cols
    spread_y = df_transformed[selected_etf] - df_transformed[prime_indicator]
    
    # 1. 원본 선 (ETF)
    fig_c.add_trace(go.Scatter(x=df_transformed.index, y=df_transformed[selected_etf], name=f"{selected_etf} 변동률(%)", line=dict(color='black', width=3)), secondary_y=False)
    
    # 2. 비교 지표 선들
    colors = ['firebrick', 'royalblue']
    for i, indicator in enumerate(top_2_indicators):
        plot_y = df_transformed[indicator]
        fig_c.add_trace(go.Scatter(x=df_transformed.index, y=plot_y, name=f"{indicator} (Lag={lag_months}M)", line=dict(color=colors[i], dash='dash')), secondary_y=(indicator in macro_cols))

    # 3. Spread (수익률 격차) Area Chart
    fig_c.add_trace(
        go.Scatter(
            x=df_transformed.index, y=spread_y, name=f"수익률(변동) 격차 vs {prime_indicator}",
            fill='tozeroy', fillcolor='rgba(128, 128, 128, 0.4)', line=dict(width=0),
            hoverinfo='y+name'
        ),
        secondary_y=False
    )

    fig_c.update_layout(
        title=f"{selected_etf} 변동률과 최상위 상관지표 변동 간 동행성 검증 (격차 색칠)",
        yaxis_title="변동률 / 격차 (%)", yaxis2_title="금리 변동 (%p)" if any(x in macro_cols for x in top_2_indicators) else "",
        height=450, hovermode="x unified"
    )
    st.plotly_chart(fig_c, use_container_width=True)

    # 하단 DART Mock 데이터 표시 (분기별 영업이익률 증감 추정)
    rep_company_map = {
        '에너지(XLE, 원유+)': '엑슨모빌(ExxonMobil), 셰브론(Chevron) 등',
        '항공(JETS, 원유-)': '델타항공, 유나이티드항공, 아메리칸항공 등',
        '금광업(GDX, 금+)': '뉴몬트, 배릭골드, 킨로스 골드 등',
        '금융(XLF, 금리+)': 'JP모건, 뱅크오브아메리카, 웰스파고 등',
        '리츠/부동산(VNQ, 금리-)': '프로로지스, 아메리칸타워, 에퀴닉스 등'
    }
    
    rep_companies = rep_company_map.get(selected_etf, '주요 상장 기업')

    st.markdown(f"**📊 해당 섹터 대표 기업 ( {rep_companies} ) 분기별 영업이익률 증감 추정치 (Mock Data)**")
    q_trend_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    
    np.random.seed(hash(selected_etf) % 2**32) 
    mock_profits = np.random.uniform(low=-5.0, high=12.0, size=len(q_trend_dates))

    fig_bar = px.bar(
        x=q_trend_dates, y=mock_profits, labels={'x': '분기', 'y': '영업이익 변동률 추정 (%)'}, text_auto=".1f"
    )
    fig_bar.update_traces(marker_color=['#2ca02c' if x >= 0 else '#d62728' for x in mock_profits])
    fig_bar.update_layout(height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ================================
    # 패널 D: 산점도(Scatter) 및 회귀선(Trendline)
    # ================================
    st.header(f"📌 패널 D: {selected_etf} 방향성 증명 산점도")
    st.markdown(f"가장 높은 상관관계를 보인 **{prime_indicator}**의 변동률(X축) 대비 **{selected_etf}**의 변동률(Y축)의 분산 및 선형추세를 확인합니다.")

    scatter_df = df_transformed[[prime_indicator, selected_etf]].dropna()
    x_val = scatter_df[prime_indicator]
    y_val = scatter_df[selected_etf]
    
    if len(scatter_df) > 1:
        m, b = np.polyfit(x_val, y_val, 1)
        trend_line = m * x_val + b

        fig_d = go.Figure()
        # 산점도
        fig_d.add_trace(go.Scatter(x=x_val, y=y_val, mode='markers', name='변동률 관측치', marker=dict(color='teal', opacity=0.6)))
        # 회귀선
        fig_d.add_trace(go.Scatter(x=x_val, y=trend_line, mode='lines', name=f'Trend Line (기울기: {m:.2f})', line=dict(color='darkred', width=3)))

        fig_d.update_layout(
            title=f"{prime_indicator} 변화에 따른 {selected_etf} 변화 추세선 검증",
            xaxis_title=f"{prime_indicator} 변동",
            yaxis_title=f"{selected_etf} 변동",
            height=450,
            hovermode="closest"
        )
        st.plotly_chart(fig_d, use_container_width=True)

st.markdown("""
> ⚠️ **분석의 한계점 및 주의사항**
> - (데이터 제약) TIGER 2차전지테마(305540) 등 상대적으로 최근에 상장된 ETF는 설정 이전의 과거 데이터(예: 5년 전)가 결측치로 처리되어 해당 기간의 상관관계 계산 모수가 적어질 수 있습니다.
> - (변동률 왜곡) 금리 등 거시지표의 변동성 폭에 비해 개별 ETF의 주가 변동성이 크므로 차트의 Y축 스케일을 유의하여 해석해야 합니다.
> - 상관성은 인과관계를 의미하지 않으며, 투자 권유를 뜻하지 않습니다.
""")
