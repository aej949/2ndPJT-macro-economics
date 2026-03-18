import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os

# 페이지 설정
st.set_page_config(page_title="금·은 자산 경제위기 분석 대시보드", layout="wide")

# 한국어 폰트 설정 (필요 시)
# st.markdown("""<style> body { font-family: 'Malgun Gothic'; } </style>""", unsafe_allow_html=True)

def load_data():
    db_path = os.path.join('gspjt', 'data', 'commodity_analysis.db')
    conn = sqlite3.connect(db_path)
    
    # 데이터 로드
    raw_df = pd.read_sql("SELECT * FROM raw_prices", conn)
    perf_df = pd.read_sql("SELECT * FROM crisis_performance", conn)
    ratio_df = pd.read_sql("SELECT * FROM gold_silver_ratio", conn)
    
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    ratio_df['Date'] = pd.to_datetime(ratio_df['Date'])
    
    conn.close()
    return raw_df, perf_df, ratio_df

def main():
    st.title("🏆 금·은 자산 경제위기 대응력 및 성과 분석")
    st.markdown("""
    본 대시보드는 지난 10년간의 주요 경제위기 국면에서 금(Gold)과 은(Silver)의 안전자산으로서의 가치를 검증합니다.
    """)

    raw_df, perf_df, ratio_df = load_data()

    # 1. 사이드바 - 위기 선택
    crises = {
        'Brexit': ('2016-06-01', '2016-08-31'),
        'Tariff War': ('2018-03-01', '2019-12-31'),
        'Iran-US Conflict': ('2020-01-01', '2020-02-29'),
        'COVID-19': ('2020-03-01', '2020-12-31'),
        'Russia-Ukraine War': ('2022-02-01', '2022-06-30')
    }

    # --- Chart 1: 전체 타임라인 및 위기 영역 ---
    st.subheader("📈 전체 타임라인: 가격 추이 및 위기 영역")
    
    # 자산별 정규화 (비교 용이성)
    raw_df['Close_Norm'] = raw_df.groupby('Ticker')['Close'].transform(lambda x: (x / x.iloc[0]) * 100)
    
    fig1 = px.line(raw_df, x='Date', y='Close_Norm', color='Asset', 
                   labels={'Close_Norm': '정규화 가격 (2015-01-01 = 100)'},
                   title="자산별 정규화 가격 추이 (2015-01-01 기준)")
    
    # 위기 영역 하이라이트
    for name, (start, end) in crises.items():
        fig1.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.1, 
                       layer="below", line_width=0,
                       annotation_text=name, annotation_position="top left")
    
    fig1.update_layout(hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    # --- Chart 2 & 4: 위기별 성과 비교 ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ 위기별 방어력 (누적 수익률)")
        fig2 = px.bar(perf_df, x='Crisis', y='Cumulative_Return', color='Asset', barmode='group',
                      labels={'Cumulative_Return': '누적 수익률'},
                      title="위기 국면별 누적 수익률 비교")
        fig2.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        st.subheader("📉 위기별 최대 낙폭 (MDD)")
        fig3 = px.bar(perf_df, x='Crisis', y='MDD', color='Asset', barmode='group',
                      labels={'MDD': '최대 낙폭 (MDD)'},
                      title="위기 국면별 최대 하락폭 비교")
        fig3.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig3, use_container_width=True)

    # --- Chart 3: 금-은 비율 추이 ---
    st.subheader("⚖️ 금-은 비율 (Gold-Silver Ratio)")
    avg_ratio = ratio_df['Ratio'].mean()
    
    fig4 = px.line(ratio_df, x='Date', y='Ratio', title="금-은 비율 변동 추이 (Gold/Silver)")
    fig4.add_hline(y=avg_ratio, line_dash="dash", line_color="red", 
                   annotation_text=f"역사적 평균: {avg_ratio:.2f}")
    st.plotly_chart(fig4, use_container_width=True)
    
    # --- 추가 분석 탭 ---
    st.subheader("📊 투자 효율성 상세 지표 (샤프 지수 & 상관계수)")
    tabs = st.tabs(["샤프 지수", "S&P 500 상관계수"])
    
    with tabs[0]:
        fig5 = px.bar(perf_df, x='Crisis', y='Sharpe_Ratio', color='Asset', barmode='group',
                      title="위기별 위험 대비 수익률 (Sharpe Ratio)")
        st.plotly_chart(fig5, use_container_width=True)
        
    with tabs[1]:
        fig6 = px.bar(perf_df, x='Crisis', y='Correlation_SP500', color='Asset', barmode='group',
                      title="S&P 500과의 상관계수 (낮을수록 안전자산 성향 강함)")
        st.plotly_chart(fig6, use_container_width=True)

    # --- 결론 및 제언 ---
    st.divider()
    st.subheader("💡 분석 결론")
    
    avg_gold_ret = perf_df[perf_df['Asset'] == 'Gold']['Cumulative_Return'].mean()
    avg_sp_ret = perf_df[perf_df['Asset'] == 'S&P500']['Cumulative_Return'].mean()
    
    st.info(f"""
    - **금의 방어적 성과**: 분석된 5대 위기 기간 동안 금은 평균적으로 S&P 500 대비 {avg_gold_ret - avg_sp_ret:+.2%}의 성과를 기록했습니다.
    - **가장 뛰어난 방어 시기**: {perf_df.loc[perf_df[perf_df['Asset']=='Gold']['Cumulative_Return'].idxmax(), 'Crisis']} 기간에 금이 가장 높은 수익률을 기록했습니다.
    - **금-은 비율 활용**: 현재 금-은 비율이 {ratio_df['Ratio'].iloc[-1]:.2f}로 역사적 평균({avg_ratio:.2f}) 대비 {'높은' if ratio_df['Ratio'].iloc[-1] > avg_ratio else '낮은'} 수준입니다.
    """)

if __name__ == "__main__":
    main()
