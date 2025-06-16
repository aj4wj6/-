from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import os
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime, timedelta
import matplotlib
from matplotlib.font_manager import fontManager
import numpy as np
from matplotlib.patches import Rectangle
import warnings
import sqlite3
import base64
from io import BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import json

warnings.filterwarnings('ignore')

# --- 字型設定 ---
FONT_PATH = os.path.join('fonts', 'NotoSansTC-Regular.ttf')
if os.path.exists(FONT_PATH):
    fontManager.addfont(FONT_PATH)
    matplotlib.rcParams['font.family'] = 'Noto Sans TC'
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    print(f"警告：找不到字型檔案 '{FONT_PATH}'。圖表中的中文可能無法正常顯示。")
    matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8')
# --- 字型設定結束 ---


app = Flask(__name__)
CORS(app)

DATABASE_PATH = 'health_reports.db'

# --- 資料庫和核心功能 (保持不變) ---
def init_database():
    """初始化資料庫"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL,
            report_data TEXT NOT NULL, pdf_path TEXT, dashboard_path TEXT,
            table_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            coach_email TEXT, patient_email TEXT, status TEXT DEFAULT 'generated'
        )''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS coaches (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL, phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL, email TEXT, phone TEXT, coach_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (coach_id) REFERENCES coaches (id)
        )''')
    conn.commit()
    conn.close()

def save_report_to_db(patient_id, report_data, pdf_path, dashboard_path, table_path, coach_email=None, patient_email=None):
    """儲存報告到資料庫"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO health_reports (patient_id, report_data, pdf_path, dashboard_path, table_path, coach_email, patient_email) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (patient_id, json.dumps(report_data), pdf_path, dashboard_path, table_path, coach_email, patient_email))
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id

def get_reports_by_patient(patient_id):
    """根據患者ID獲取報告"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM health_reports WHERE patient_id = ? ORDER BY created_at DESC', (patient_id,))
    reports = cursor.fetchall()
    conn.close()
    return reports

def send_email_report(sender_email, sender_password, recipient_email, patient_id, pdf_path, report_type="patient"):
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        if report_type == "coach":
            msg['Subject'] = f"學員 {patient_id} 的健康數據報告"
            body = f"親愛的教練：\n\n您好！這是學員 {patient_id} 的最新健康數據報告。\n\n請查看附件。\n\n健身數據分析系統"
        else:
            msg['Subject'] = f"您的個人健康數據報告 - {patient_id}"
            body = f"親愛的會員：\n\n您好！這是您的最新健康數據分析報告。\n\n請查看附件。\n\n健身數據分析系統"
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(pdf_path)}"')
                msg.attach(part)
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return True, "郵件發送成功"
    except Exception as e:
        if isinstance(e, smtplib.SMTPAuthenticationError): return False, "SMTP驗證錯誤，請檢查寄件Email和應用程式密碼。"
        return False, f"郵件發送失敗: {str(e)}"

# --- 視覺化 Class (保持不變) ---
class HealthDataVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.colors = {
            'primary': '#2E86AB', 'secondary': '#A23B72', 'success': '#F18F01',
            'warning': '#C73E1D', 'info': '#6A994E'
        }
    
    def create_dashboard_chart(self, data):
        plt.rcParams['font.family'] = 'Noto Sans TC'
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'{data["patient_id"]} 健康數據儀表板', fontsize=20, fontweight='bold')
        gs = fig.add_gridspec(2, 3)
        ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
        ax4, ax5, ax6 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2], polar=True)
        self._create_gauge_chart(ax1, data['heart_rate'], 'heart_rate', '心率 (BPM)')
        self._create_bmi_chart(ax2, data['bmi'], data['weight'], data['height'])
        self._create_blood_pressure_chart(ax3, data['blood_pressure'])
        self._create_exercise_chart(ax4, data['exercise_duration'])
        self._create_trend_chart(ax5, data)
        self._create_health_score_chart(ax6, data)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        dashboard_path = os.path.join(self.output_dir, f'dashboard_{data["patient_id"]}_{timestamp}.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return dashboard_path
    
    def _create_gauge_chart(self, ax, value, metric_type, title):
        ranges, max_val = [(0,60,'#FF6B6B'),(60,100,'#4ECDC4'),(100,160,'#FFE66D'),(160,220,'#FF6B6B')], 220
        theta = np.linspace(0, np.pi, 200)
        for s,e,c in ranges:
            m = (theta >= s/max_val*np.pi) & (theta <= e/max_val*np.pi)
            if np.any(m): ax.fill_between(theta[m], 0.8, 1.0, color=c, alpha=0.7)
        if value is not None and value > 0:
            a = min(value/max_val*np.pi, np.pi)
            ax.plot([a,a],[0,0.9],'k-',lw=4); ax.plot(a,0,'ko',ms=8)
            ax.text(np.pi/2,0.5,f'{int(value)}',ha='center',va='center',fontsize=16,fontweight='bold')
        else: ax.text(np.pi/2,0.5,'N/A',ha='center',va='center',fontsize=16,fontweight='bold')
        ax.set_xlim(0,np.pi); ax.set_ylim(0,1); ax.set_title(title,fontsize=14,fontweight='bold'); ax.axis('off')

    def _create_bmi_chart(self, ax, bmi, weight, height):
        cats, colors = ['過輕','正常','過重','肥胖'], ['#74C0FC','#51CF66','#FFD43B','#FF6B6B']
        widths, starts = [18.5,5.5,3,23], [0,18.5,24,27]
        y_pos = np.arange(len(cats))
        ax.barh(y_pos, widths, left=starts, color=colors, alpha=0.7, height=0.6, align='center')
        if bmi is not None and bmi > 0:
            ax.axvline(x=bmi,c='r',ls='--',lw=3,label=f'您的 BMI: {bmi:.1f}')
            ax.text(bmi,len(cats)-0.5,f'{bmi:.1f}',ha='center',va='bottom',fontweight='bold',fontsize=12,color='r')
        ax.set_yticks(y_pos); ax.set_yticklabels(cats); ax.set_xlabel('BMI 數值')
        ax.set_title('BMI 分析圖',fontsize=14,fontweight='bold'); ax.legend(); ax.grid(axis='x',alpha=0.3); ax.set_xlim(10,40)

    def _create_blood_pressure_chart(self, ax, blood_pressure):
        if not blood_pressure or '/' not in str(blood_pressure):
            ax.text(0.5,0.5,'血壓數據無效',ha='center',va='center',transform=ax.transAxes,fontsize=14); ax.set_title('血壓分析',fontsize=14,fontweight='bold'); return
        try: sys, dia = map(int, str(blood_pressure).split('/'))
        except (ValueError,TypeError):
            ax.text(0.5,0.5,'血壓格式錯誤',ha='center',va='center',transform=ax.transAxes,fontsize=14); ax.set_title('血壓分析',fontsize=14,fontweight='bold'); return
        cats = {'理想':{'sys':(0,120),'dia':(0,80),'c':'#51CF66'}, '正常':{'sys':(120,130),'dia':(80,85),'c':'#A9E0A9'}, '偏高':{'sys':(130,140),'dia':(85,90),'c':'#FFD43B'}, '高血壓':{'sys':(140,200),'dia':(90,120),'c':'#FF6B6B'}}
        for cat, r in cats.items(): ax.add_patch(Rectangle((r['sys'][0],r['dia'][0]),r['sys'][1]-r['sys'][0],r['dia'][1]-r['dia'][0],alpha=0.3,color=r['c'],label=cat))
        ax.scatter(sys,dia,s=200,c='r',marker='*',edgecolors='k',lw=2,label=f'您的血壓: {blood_pressure}')
        ax.set_xlabel('收縮壓 (mmHg)'); ax.set_ylabel('舒張壓 (mmHg)'); ax.set_title('血壓分析圖',fontsize=14,fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05,1),loc='upper left'); ax.grid(True,alpha=0.3); ax.set_xlim(80,200); ax.set_ylim(50,120)

    def _create_exercise_chart(self, ax, exercise_duration):
        recs = {'最低標準':150, '理想目標':300, '當週運動':exercise_duration or 0}
        cats, vals = list(recs.keys()), list(recs.values())
        bars = ax.bar(cats, vals, color=['#FFD43B','#4ECDC4','#2E86AB'], alpha=0.8)
        for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()+5,f'{int(val)} 分鐘',ha='center',va='bottom',fontweight='bold')
        ax.set_ylabel('運動時間 (分鐘)'); ax.set_title('每週運動時間分析',fontsize=14,fontweight='bold'); ax.grid(axis='y',alpha=0.3)
        ax.axhline(150,c='r',ls='--',alpha=0.7,label='WHO 最低建議'); ax.axhline(300,c='g',ls='--',alpha=0.7,label='WHO 理想目標'); ax.legend()

    def _create_trend_chart(self, ax, data):
        dates = [datetime.now()-timedelta(days=x) for x in range(30,0,-1)]
        hr = data.get('heart_rate') or 75; hr_trend = [max(50,hr+np.random.normal(0,5)) for _ in dates]
        w = data.get('weight') or 70; w_trend = [max(40,w+np.random.normal(0,0.5)) for _ in dates]
        ax2 = ax.twinx()
        l1 = ax.plot(dates,hr_trend,c=self.colors['primary'],lw=2,marker='o',ms=3,label='心率')
        l2 = ax2.plot(dates,w_trend,c=self.colors['secondary'],lw=2,marker='s',ms=3,label='體重')
        ax.set_xlabel('日期'); ax.set_ylabel('心率 (BPM)',c=self.colors['primary']); ax2.set_ylabel('體重 (kg)',c=self.colors['secondary'])
        ax.set_title('30 天健康趨勢',fontsize=14,fontweight='bold'); lines=l1+l2; ax.legend(lines,[l.get_label() for l in lines],loc='upper left')
        ax.grid(True,alpha=0.3); ax.tick_params(axis='x',rotation=45)

    def _create_health_score_chart(self, ax, data):
        scores = {}
        hr = data.get('heart_rate'); scores['心率'] = 100 if 60<=hr<=80 else 80 if 50<=hr<=100 else 60 if hr else 0
        bmi = data.get('bmi'); scores['BMI'] = 100 if 18.5<=bmi<24 else 70 if 24<=bmi<27 or 17<=bmi<18.5 else 50 if bmi else 0
        scores['運動'] = min(100, ((data.get('exercise_duration') or 0) / 300) * 100)
        try: sys,dia = map(int, str(data.get('blood_pressure')).split('/')); scores['血壓'] = 100 if sys<120 and dia<80 else 70 if sys<140 and dia<90 else 50
        except: scores['血壓'] = 0
        cats, vals = list(scores.keys()), list(scores.values())
        if not cats: ax.text(0.5,0.5,'無足夠數據評分',ha='center',va='center'); ax.set_title('綜合健康評分',fontsize=14,fontweight='bold'); ax.axis('off'); return
        angles = np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist()
        vals_c, angles_c = vals+vals[:1], angles+angles[:1]
        ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
        ax.plot(angles_c,vals_c,'o-',lw=2,color=self.colors['info'],zorder=3); ax.fill(angles_c,vals_c,alpha=0.25,color=self.colors['info'],zorder=2)
        ax.set_thetagrids(np.degrees(angles),cats); ax.set_rgrids([20,40,60,80,100]); ax.set_ylim(0,100)
        ax.set_title('綜合健康評分',fontsize=14,fontweight='bold',y=1.1)
        score = sum(vals)/len(vals) if vals else 0
        ax.text(0,0,f'綜合評分\n{score:.1f}',ha='center',va='center',fontsize=14,fontweight='bold',bbox=dict(boxstyle="round,pad=0.3",facecolor="yellow",alpha=0.5),zorder=4)

# --- 報告生成邏輯 (保持不變) ---
def _generate_recommendations(data):
    recs = []
    hr = data.get('heart_rate'); recs.append(f"1. 您的靜息心率{'偏高...' if hr > 100 else '偏低...' if hr < 60 else '在理想範圍' if hr else '未提供'}")
    bmi = data.get('bmi'); recs.append(f"2. 您的BMI{'過低...' if bmi < 18.5 else '正常' if bmi < 24 else '過重...' if bmi < 27 else '肥胖...' if bmi else '未提供'}")
    try: sys,dia=map(int,str(data.get('blood_pressure')).split('/')); recs.append(f"3. 您的血壓{'理想' if sys<120 and dia<80 else '偏高...' if sys<140 and dia<90 else '高血壓...'}")
    except: recs.append("3. 血壓資料未提供或格式錯誤")
    ex = data.get('exercise_duration') or 0; recs.append(f"4. 本週運動時間({int(ex)}分鐘){'不足...' if ex < 150 else '達標' if ex <= 300 else '充足...'}")
    recs.append("5. 通用建議：規律作息，均衡飲食。")
    return recs

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        patient_id, coach_email, patient_email = data.get('patient_id','未知'), data.get('coach_email'), data.get('patient_email')
        send_email, sender_email, sender_password = data.get('send_email',False), data.get('sender_email'), data.get('sender_password')
        if send_email and (not sender_email or not sender_password):
            return jsonify({"status":"error", "message":"需提供寄件Email和密碼"}), 400
        def safe_float(v): return float(v) if v is not None else None
        p_data = {"patient_id":patient_id, "heart_rate":safe_float(data.get('heart_rate')), "weight":safe_float(data.get('weight')), "height":safe_float(data.get('height')), "bmi":safe_float(data.get('bmi')), "blood_pressure":data.get('blood_pressure'), "exercise_duration":safe_float(data.get('exercise_duration'))}
        output_dir='output'; os.makedirs(output_dir,exist_ok=True)
        visualizer = HealthDataVisualizer(output_dir=output_dir)
        dashboard_path = visualizer.create_dashboard_chart(p_data)
        recs = _generate_recommendations(p_data)
        class SimplePDF(FPDF):
            def header(self): self.set_font('NotoSansTC','B',16); self.cell(0,10,f'個人化健康報告 - {patient_id}',0,1,'C'); self.set_font('NotoSansTC','',10); self.cell(0,8,f'報告生成時間：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',0,1,'C'); self.ln(10)
            def footer(self): self.set_y(-15); self.set_font('NotoSansTC','',8); self.cell(0,10,f'第 {self.page_no()} 頁',0,0,'C')
        pdf = SimplePDF()
        if not os.path.exists(FONT_PATH): return jsonify({"status":"error", "message":f"字體檔案 '{FONT_PATH}' 不存在"}), 500
        pdf.add_font('NotoSansTC','',FONT_PATH,uni=True); pdf.add_font('NotoSansTC','B',FONT_PATH,uni=True)
        pdf.add_page(); 
        pdf.set_font('NotoSansTC','B',14); 
        pdf.cell(0,10,'您的個人化健康建議',ln=1);
        pdf.set_font('NotoSansTC','',12); pdf.ln(5)
        for r in recs: pdf.multi_cell(0,8,r,border=0); pdf.ln(2)
        pdf.add_page(); pdf.set_font('NotoSansTC','B',14); pdf.cell(0,10,'附錄：健康數據視覺化圖表',ln=1,align='C'); pdf.ln(5)
        if os.path.exists(dashboard_path): pdf.image(dashboard_path,x=10,w=190)
        report_path = os.path.join(output_dir,f'health_report_{patient_id}_{datetime.now().strftime("%Y%m%d%H%M%S")}.pdf')
        pdf.output(report_path)
        report_id = save_report_to_db(patient_id,p_data,report_path,dashboard_path,None,coach_email,patient_email)
        res_data = {'status':'success','message':'報告生成成功','report_id':report_id,'patient_name':patient_id,'view_url':f'/view_report/{report_id}','download_url':f'/download_report/{report_id}'}
        if send_email:
            email_res = []
            if coach_email: success,msg=send_email_report(sender_email,sender_password,coach_email,patient_id,report_path,"coach"); email_res.append({"recipient":"coach","status":success,"message":msg})
            if patient_email: success,msg=send_email_report(sender_email,sender_password,patient_email,patient_id,report_path,"patient"); email_res.append({"recipient":"patient","status":success,"message":msg})
            res_data['email_results'] = email_res
        return jsonify(res_data), 200
    except Exception as e:
        traceback_str = traceback.format_exc(); print(traceback_str)
        return jsonify({"status":"error","message":str(e),"traceback":traceback_str}), 500


# --- 路由 (修改 HTML 格式) ---
@app.route('/view_report/<int:report_id>')
def view_report(report_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM health_reports WHERE id = ?', (report_id,))
    report = cursor.fetchone()
    conn.close()
    if not report:
        return "報告不存在", 404
    
    dashboard_base64 = ""
    report_data = json.loads(report[2])
    if report[4] and os.path.exists(report[4]):
        with open(report[4], 'rb') as img_file:
            dashboard_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    # --- HTML 模板整理開始 ---
    html_template = """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>健康報告 - {{ patient_id }}</title>
        <style>
            body { 
                font-family: 'Microsoft JhengHei', sans-serif; 
                margin: 20px; 
                background: #f0f2f5; 
            }
            .container { 
                max-width: 1200px; 
                margin: auto; 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,.1); 
                overflow: hidden; 
            }
            .header { 
                background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%); 
                color: white; 
                padding: 30px; 
                text-align: center; 
            }
            .content { 
                padding: 30px; 
            }
            .report-info { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px; 
            }
            .info-card { 
                background: #f8f9fa; 
                border-radius: 10px; 
                padding: 20px; 
                border-left: 5px solid #2E86AB; 
            }
            .info-card h3 { 
                margin-top: 0; 
            }
            .info-card p { 
                font-size: 1.5em; 
                font-weight: 700; 
                margin: 10px 0 0; 
            }
            .dashboard-container img { 
                max-width: 100%; 
                border-radius: 10px; 
            }
            .actions { 
                text-align: center; 
                padding: 20px; 
                background: #f8f9fa; 
            }
            .btn { 
                display: inline-block; 
                padding: 12px 30px; 
                margin: 10px; 
                background: #2E86AB; 
                color: white; 
                text-decoration: none; 
                border-radius: 25px; 
                transition: .3s ease; 
            }
            .btn:hover { 
                background: #1e5f7a; 
            }
            .btn-secondary { 
                background: #6c757d; 
            }
            .btn-secondary:hover { 
                background: #545b62; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏃‍♂️ 健康數據報告</h1>
                <p>會員ID: {{ patient_id }}</p>
                <div>生成時間: {{ created_at }}</div>
            </div>
            <div class="content">
                <div class="report-info">
                    <div class="info-card"><h3>❤️ 心率</h3><p>{{ heart_rate | default('N/A') }} 次/分</p></div>
                    <div class="info-card"><h3>⚖️ 體重</h3><p>{{ weight | default('N/A') }} 公斤</p></div>
                    <div class="info-card"><h3>📏 身高</h3><p>{{ height | default('N/A') }} 公分</p></div>
                    <div class="info-card"><h3>📊 BMI</h3><p>{{ bmi | default('N/A') }}</p></div>
                    <div class="info-card"><h3>🩸 血壓</h3><p>{{ blood_pressure | default('N/A') }} mmHg</p></div>
                    <div class="info-card"><h3>🏃‍♀️ 運動時間</h3><p>{{ exercise_duration | default('N/A') }} 分鐘</p></div>
                </div>
                <div class="dashboard-container">
                    <h2>📈 健康數據儀表板</h2>
                    {% if dashboard_image %}
                        <img src="data:image/png;base64,{{ dashboard_image }}">
                    {% else %}
                        <p>儀表板圖片載入失敗</p>
                    {% endif %}
                </div>
            </div>
            <div class="actions">
                <a href="/download_report/{{ report_id }}" class="btn">📥 下載PDF</a>
                <a href="/reports/{{ patient_id }}" class="btn btn-secondary">📋 歷史報告</a>
            </div>
        </div>
    </body>
    </html>
    """
    # --- HTML 模板整理結束 ---
    
    return render_template_string(html_template, 
        report_id=report_id, 
        created_at=report[6], 
        dashboard_image=dashboard_base64, 
        **report_data
    )

@app.route('/download_report/<int:report_id>')
def download_report(report_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT pdf_path FROM health_reports WHERE id = ?', (report_id,))
    result = cursor.fetchone()
    conn.close()
    if not result or not result[0] or not os.path.exists(result[0]):
        return "報告檔案不存在或路徑已失效", 404
    return send_file(result[0], as_attachment=True)

@app.route('/reports/<patient_id>')
def list_reports(patient_id):
    reports = get_reports_by_patient(patient_id)
    def format_dt(ts_str, fmt):
        try:
            return datetime.fromisoformat(ts_str.split('.')[0]).strftime(fmt)
        except:
            return ts_str
            
    # --- HTML 模板整理開始 ---
    html_template = """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>歷史報告 - {{ patient_id }}</title>
        <style>
            body { 
                font-family: 'Microsoft JhengHei', sans-serif; 
                padding: 20px; 
                background: #f5f5f5; 
            }
            .container { 
                max-width: 1000px; 
                margin: auto; 
                background: white; 
                border-radius: 10px; 
                box-shadow: 0 5px 15px rgba(0,0,0,.1); 
                overflow: hidden; 
            }
            .header { 
                background: #2E86AB; 
                color: white; 
                padding: 20px; 
                text-align: center; 
            }
            .report-list { 
                padding: 20px; 
            }
            .report-item { 
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                margin-bottom: 15px; 
                padding: 15px; 
                background: #f9f9f9; 
                transition: .3s ease; 
            }
            .report-item:hover { 
                background: #e3f2fd; 
                transform: translateY(-2px); 
            }
            .report-date { 
                font-weight: 700; 
                color: #2E86AB; 
            }
            .btn { 
                display: inline-block; 
                padding: 8px 16px; 
                margin-left: 10px; 
                background: #2E86AB; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                font-size: 14px; 
                transition: .3s ease; 
            }
            .btn:hover { 
                background: #1e5f7a; 
            }
            .btn-secondary { 
                background: #6c757d; 
            }
            .btn-secondary:hover { 
                background: #545b62; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📋 {{ patient_id }} 的歷史報告</h1>
                <p>共 {{ reports|length }} 份報告</p>
            </div>
            <div class="report-list">
                {% for report in reports %}
                    <div class="report-item">
                        <div class="report-date">📅 {{ format_dt(report[6], '%Y-%m-%d %H:%M') }}</div>
                        <div>
                            <a href="/view_report/{{ report[0] }}" class="btn">👀 查看</a>
                            <a href="/download_report/{{ report[0] }}" class="btn btn-secondary">📥 下載</a>
                        </div>
                    </div>
                {% else %}
                    <p style="text-align:center; color:#666; margin:50px 0;">📭 尚無報告記錄</p>
                {% endfor %}
                <div style="text-align:center; margin-top:30px;">
                    <a href="/" class="btn">🏠 返回首頁</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    # --- HTML 模板整理結束 ---
    
    return render_template_string(html_template, 
        patient_id=patient_id, 
        reports=reports, 
        format_dt=format_dt
    )

@app.route('/api/coaches', methods=['GET', 'POST'])
def manage_coaches():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    if request.method == 'GET':
        cursor.execute('SELECT id, name, email, phone, created_at FROM coaches ORDER BY name')
        coaches = cursor.fetchall()
        conn.close()
        return jsonify([dict(zip(['id','name','email','phone','created_at'], row)) for row in coaches])
    if request.method == 'POST':
        data = request.get_json()
        if not data or not data.get('name') or not data.get('email'):
            return jsonify({'error': '姓名和Email為必填項'}), 400
        try:
            cursor.execute('INSERT INTO coaches (name, email, phone) VALUES (?, ?, ?)', (data.get('name'), data.get('email'), data.get('phone')))
            conn.commit()
            coach_id = cursor.lastrowid
            return jsonify({'id': coach_id, 'message': '教練新增成功'}), 201
        except sqlite3.IntegrityError:
            return jsonify({'error': '此Email已被註冊'}), 409
        finally:
            conn.close()

if __name__ == '__main__':
    init_database()
    app.run(host='127.0.0.1', debug=True, port=5000)