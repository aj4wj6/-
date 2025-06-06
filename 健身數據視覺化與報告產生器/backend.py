from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime, timedelta
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# 支援中文顯示（僅影響 matplotlib 圖表）
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)  # 允許跨域資源共享

class HealthDataVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6A994E'
        }
    
    def create_dashboard_chart(self, data):
        """創建綜合儀表板圖表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{data["patient_id"]} 健康數據儀表板', fontsize=20, fontweight='bold')
        
        # 1. 心率儀表
        self._create_gauge_chart(axes[0, 0], data['heart_rate'], 'heart_rate', '心率 (BPM)')
        
        # 2. BMI分析
        self._create_bmi_chart(axes[0, 1], data['bmi'], data['weight'], data['height'])
        
        # 3. 血壓分析
        self._create_blood_pressure_chart(axes[0, 2], data['blood_pressure'])
        
        # 4. 運動時間分析
        self._create_exercise_chart(axes[1, 0], data['exercise_duration'])
        
        # 5. 健康趨勢模擬（假設有歷史數據）
        self._create_trend_chart(axes[1, 1], data)
        
        # 6. 綜合健康評分
        self._create_health_score_chart(axes[1, 2], data)
        
        plt.tight_layout()
        dashboard_path = os.path.join(self.output_dir, 'health_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dashboard_path
    
    def _create_gauge_chart(self, ax, value, metric_type, title):
        """創建儀表盤圖表"""
        if metric_type == 'heart_rate':
            ranges = [(0, 60, '#FF6B6B'), (60, 100, '#4ECDC4'), (100, 160, '#FFE66D'), (160, 220, '#FF6B6B')]
        
        theta = np.linspace(0, np.pi, 100)
        for start, end, color in ranges:
            start_angle = start / 220 * np.pi
            end_angle = end / 220 * np.pi
            mask = (theta >= start_angle) & (theta <= end_angle)
            if np.any(mask):
                ax.fill_between(theta[mask], 0.8, 1.0, color=color, alpha=0.7)
        
        if value and value > 0:
            needle_angle = min(value / 220 * np.pi, np.pi)
            ax.plot([needle_angle, needle_angle], [0, 0.9], 'k-', linewidth=4)
            ax.plot(needle_angle, 0, 'ko', markersize=8)
            ax.text(np.pi/2, 0.5, f'{value}', ha='center', va='center', fontsize=16, fontweight='bold')
        else:
            ax.text(np.pi/2, 0.5, 'N/A', ha='center', va='center', fontsize=16, fontweight='bold')
        
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _create_bmi_chart(self, ax, bmi, weight, height):
        """創建 BMI 分析圖表"""
        bmi_categories = ['過輕', '正常', '過重', '肥胖']
        colors = ['#74C0FC', '#51CF66', '#FFD43B', '#FF6B6B']
        
        y_pos = np.arange(len(bmi_categories))
        bars = ax.barh(y_pos, [18.5, 5.5, 3, 23], left=[0, 18.5, 24, 27],
                       color=colors, alpha=0.7, height=0.6)
        
        if bmi and bmi > 0:
            ax.axvline(x=bmi, color='red', linestyle='--', linewidth=3, label=f'您的 BMI: {bmi:.1f}')
            ax.text(bmi, len(bmi_categories), f'{bmi:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bmi_categories)
        ax.set_xlabel('BMI 數值')
        ax.set_title('BMI 分析圖', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 50)
    
    def _create_blood_pressure_chart(self, ax, blood_pressure):
        """創建血壓分析圖表"""
        if not blood_pressure or '/' not in str(blood_pressure):
            ax.text(0.5, 0.5, '血壓數據無效', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('血壓分析', fontsize=14, fontweight='bold')
            return
        
        try:
            systolic, diastolic = map(int, str(blood_pressure).split('/'))
        except:
            ax.text(0.5, 0.5, '血壓格式錯誤', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('血壓分析', fontsize=14, fontweight='bold')
            return
        
        bp_categories = {
            '理想':    {'sys': (0, 120),  'dia': (0, 80),  'color': '#51CF66'},
            '正常':    {'sys': (120, 130),'dia': (80, 85), 'color': '#FFD43B'},
            '偏高':    {'sys': (130, 140),'dia': (85, 90), 'color': '#FF8C42'},
            '高血壓':  {'sys': (140, 200),'dia': (90, 120),'color': '#FF6B6B'}
        }
        
        for category, ranges in bp_categories.items():
            rect = Rectangle((ranges['sys'][0], ranges['dia'][0]),
                             ranges['sys'][1] - ranges['sys'][0],
                             ranges['dia'][1] - ranges['dia'][0],
                             alpha=0.3, color=ranges['color'], label=category)
            ax.add_patch(rect)
        
        ax.scatter(systolic, diastolic, s=200, c='red', marker='*',
                   edgecolors='black', linewidth=2, label=f'您的血壓: {blood_pressure}')
        
        ax.set_xlabel('收縮壓 (mmHg)')
        ax.set_ylabel('舒張壓 (mmHg)')
        ax.set_title('血壓分析圖', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(80, 200)
        ax.set_ylim(50, 120)
    
    def _create_exercise_chart(self, ax, exercise_duration):
        """創建運動時間分析圖表"""
        recommendations = {
            '最低標準':  150,
            '理想目標':  300,
            '當週運動':  exercise_duration or 0
        }
        categories = list(recommendations.keys())
        values = list(recommendations.values())
        colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{value} 分鐘', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('運動時間 (分鐘)')
        ax.set_title('每週運動時間分析', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=150, color='red', linestyle='--', alpha=0.7, label='WHO 最低建議')
        ax.axhline(y=300, color='green', linestyle='--', alpha=0.7, label='WHO 理想目標')
        ax.legend()
    
    def _create_trend_chart(self, ax, data):
        """創建健康趨勢圖（模擬數據）"""
        dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
        base_hr = data.get('heart_rate', 75) or 75
        hr_trend = [max(50, base_hr + np.random.normal(0, 5)) for _ in dates]
        base_weight = data.get('weight', 70) or 70
        weight_trend = [max(40, base_weight + np.random.normal(0, 0.5)) for _ in dates]
        
        ax2 = ax.twinx()
        line1 = ax.plot(dates, hr_trend, color=self.colors['primary'],
                        linewidth=2, marker='o', markersize=3, label='心率')
        line2 = ax2.plot(dates, weight_trend, color=self.colors['secondary'],
                         linewidth=2, marker='s', markersize=3, label='體重')
        
        ax.set_xlabel('日期')
        ax.set_ylabel('心率 (BPM)', color=self.colors['primary'])
        ax2.set_ylabel('體重 (kg)', color=self.colors['secondary'])
        ax.set_title('30 天健康趨勢', fontsize=14, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _create_health_score_chart(self, ax, data):
        """創建綜合健康評分圖表"""
        scores = {}
        hr = data.get('heart_rate')
        if hr and hr > 0:
            if 60 <= hr <= 80:
                scores['心率'] = 100
            elif 50 <= hr <= 100:
                scores['心率'] = 80
            else:
                scores['心率'] = 60
        else:
            scores['心率'] = 0
        
        bmi = data.get('bmi')
        if bmi and bmi > 0:
            if 18.5 <= bmi < 24:
                scores['BMI'] = 100
            elif 24 <= bmi < 27 or 17 <= bmi < 18.5:
                scores['BMI'] = 70
            else:
                scores['BMI'] = 50
        else:
            scores['BMI'] = 0
        
        exercise = data.get('exercise_duration') or 0
        if exercise >= 300:
            scores['運動'] = 100
        elif exercise >= 150:
            scores['運動'] = 80
        elif exercise >= 75:
            scores['運動'] = 60
        else:
            scores['運動'] = 40
        
        bp = data.get('blood_pressure')
        if bp and '/' in str(bp):
            try:
                sys, dia = map(int, str(bp).split('/'))
                if sys < 120 and dia < 80:
                    scores['血壓'] = 100
                elif sys < 140 and dia < 90:
                    scores['血壓'] = 70
                else:
                    scores['血壓'] = 50
            except:
                scores['血壓'] = 0
        else:
            scores['血壓'] = 0
        
        categories = list(scores.keys())
        values = list(scores.values())
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['info'])
        ax.fill(angles, values, alpha=0.25, color=self.colors['info'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_title('綜合健康評分', fontsize=14, fontweight='bold')
        ax.grid(True)
        
        total_score = sum(values[:-1]) / len(values[:-1]) if len(values) > 1 else 0
        ax.text(0, -20, f'綜合評分: {total_score:.1f}/100',
                ha='center', va='center', transform=ax.transData,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))


def _generate_recommendations(data):
    """根據輸入健康數據生成個人化建議"""
    recommendations = []
    
    # 心率建議
    heart_rate = data.get('heart_rate')
    if heart_rate and heart_rate > 0:
        if heart_rate > 100:
            recommendations.append("1. 靜息心率偏高，建議：進行放鬆練習（深呼吸、冥想）以降低靜息心率。")
        elif heart_rate < 60:
            recommendations.append("1. 靜息心率偏低，若非運動員，建議：諮詢醫生確認是否有心臟功能異常。")
        else:
            recommendations.append("1. 心率正常，請持續維持目前生活方式。")
    else:
        recommendations.append("1. 未提供心率資料，建議：定期測量並輸入心率資訊。")
    
    # BMI 建議
    bmi = data.get('bmi')
    if bmi and bmi > 0:
        if bmi < 18.5:
            recommendations.append("2. BMI 過低，建議：提高蛋白質攝取並進行阻力訓練以健康增重。")
        elif bmi < 24:
            recommendations.append("2. BMI 正常，請持續保持均衡飲食與規律運動。")
        elif bmi < 27:
            recommendations.append("2. BMI 過重，建議：控制飲食熱量並增加有氧運動頻率。")
        else:
            recommendations.append("2. BMI 肥胖，強烈建議：諮詢營養師與醫師，制定個人化減重計畫。")
    else:
        recommendations.append("2. 未提供 BMI 資料，建議：輸入身高體重，系統自動計算 BMI。")
    
    # 血壓建議
    blood_pressure = data.get('blood_pressure')
    if blood_pressure and '/' in str(blood_pressure):
        try:
            sys, dia = map(int, str(blood_pressure).split('/'))
            if sys < 120 and dia < 80:
                recommendations.append("3. 血壓理想，請持續保持健康生活方式。")
            elif sys < 140 and dia < 90:
                recommendations.append("3. 血壓偏高，建議：減少鹽分攝取、增加運動、管理壓力。")
            else:
                recommendations.append("3. 血壓高血壓，強烈建議：儘速諮詢醫師，可能需要藥物治療。")
        except:
            recommendations.append("3. 血壓格式錯誤，建議：確認輸入格式為 “收縮壓/舒張壓”。")
    else:
        recommendations.append("3. 未提供血壓資料，建議：定期測量並輸入血壓資訊。")
    
    # 運動時間建議
    exercise_duration = data.get('exercise_duration') or 0
    if exercise_duration < 75:
        recommendations.append("4. 本週運動時間嚴重不足，建議：逐步增加至每週150分鐘中等強度運動。")
    elif exercise_duration < 150:
        recommendations.append("4. 本週運動時間不足，建議：再增加50–75分鐘即可達到 WHO 最低標準。")
    elif exercise_duration <= 300:
        recommendations.append("4. 本週運動時間已達標，請持續保持並注意適度休息。")
    else:
        recommendations.append("4. 本週運動量充足，請注意適度休息以避免過度訓練。")
    
    # 通用建議
    recommendations.append("5. 保持規律作息，每晚 7–9 小時充足睡眠。")
    recommendations.append("6. 多攝取蔬果、全穀類與優質蛋白質。")
    recommendations.append("7. 定期監測健康數據，建立健康日記以追蹤變化。")
    recommendations.append("8. 保持心情愉快，適當管理壓力。")
    
    return recommendations

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()

        patient_id = data.get('patient_id', '未知')
        heart_rate = data.get('heart_rate')
        weight = data.get('weight')
        height = data.get('height')
        bmi = data.get('bmi')
        blood_pressure = data.get('blood_pressure')
        exercise_duration = data.get('exercise_duration')

        # 數據驗證與轉換
        if heart_rate:
            heart_rate = float(heart_rate)
        if weight:
            weight = float(weight)
        if height:
            height = float(height)
        if bmi:
            bmi = float(bmi)
        if exercise_duration:
            exercise_duration = float(exercise_duration)

        patient_data = {
            "patient_id": patient_id,
            "heart_rate": heart_rate,
            "bmi": bmi,
            "weight": weight,
            "height": height,
            "blood_pressure": blood_pressure,
            "exercise_duration": exercise_duration
        }

        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        visualizer = HealthDataVisualizer(output_dir=output_dir)
        dashboard_path = visualizer.create_dashboard_chart(patient_data)

        # 取得個人化建議
        recommendations = _generate_recommendations(patient_data)

        # 開始產生 PDF
        class SimplePDF(FPDF):
            def header(self):
                # PDF 標頭，使用註冊後的 NotoSansTC
                self.set_font('NotoSansTC', 'B', 16)
                self.cell(0, 10, '健康報告 建議清單', 0, 1, 'C')
                self.ln(5)

        pdf = SimplePDF()
        # 先註冊中文字型，family 名稱為 "NotoSansTC"
        pdf.add_font('NotoSansTC', '',  'fonts/NotoSansTC-Regular.ttf', uni=True)
        pdf.add_font('NotoSansTC', 'B', 'fonts/NotoSansTC-Regular.ttf', uni=True)
        pdf.add_page()

        # 第一頁：個人化建議
        pdf.set_font('NotoSansTC', '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        pdf.cell(0, 8, f'患者 ID：{patient_id}', ln=1)
        pdf.cell(0, 8, f'報告生成時間：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=1)
        pdf.ln(5)
        
        for rec in recommendations:
            pdf.multi_cell(0, 8, rec, border=0)
            pdf.ln(1)

        # 第二頁：附錄放置圖表（選擇性）
        pdf.add_page()
        pdf.set_font('NotoSansTC', 'B', 14)
        pdf.cell(0, 10, '附錄：健康數據圖表', ln=1, align='C')
        pdf.ln(5)

        # 若有表格截圖，放在此處
        table_path = os.path.join(output_dir, 'health_data_table.png')
        if os.path.exists(table_path):
            pdf.image(table_path, x=10, w=190)
            pdf.ln(5)

        if os.path.exists(dashboard_path):
            pdf.image(dashboard_path, x=10, w=190)

        report_path = os.path.join(
            output_dir,
            f'health_report_{patient_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
        pdf.output(report_path)

        return jsonify({
            "status": "success",
            "message": "健康報告生成成功",
            "report_path": report_path,
            "patient_name": patient_id,
            "dashboard_path": dashboard_path
        }), 200

    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback_str
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=5000)
