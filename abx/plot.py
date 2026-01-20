import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joypy
import glob
import os
from matplotlib import cm
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# --- 1. 全局配置 ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.weight": "normal",
    "axes.labelweight": "normal"
})

# --- 2. 变量定义 (区分匹配名与显示名) ---
# 匹配名：必须和 CSV 文件里的字符串一致（不区分大小写）
match_name_B = "lemas-edit"

# 显示名：你在 PDF 图表底部想看到的名称
model_A = "LEMAS-TTS"
model_B = "LEMAS-Edit"

# --- 3. 数据处理 ---
input_folder = './res'  
all_files = glob.glob(os.path.join(input_folder, "*.csv"))
df = pd.concat([pd.read_csv(f) for f in all_files], axis=0, ignore_index=True)

def normalize_to_B(row):
    # 使用 .lower() 强制转换，防止大小写导致找不到数据
    cond_a_val = str(row['cond_A']).lower()
    cond_b_val = str(row['cond_B']).lower()
    
    if cond_b_val == match_name_B:
        return row['pref']
    elif cond_a_val == match_name_B:
        return 100 - row['pref']
    return np.nan

df['norm_score'] = df.apply(normalize_to_B, axis=1)

# 检查是否匹配成功
if df['norm_score'].isnull().all():
    print("Error: 匹配失败！请检查 CSV 中的 cond_A/B 列是否包含 'lemas-edit'")
else:
    df['lang_id'] = df['sample'].apply(lambda x: str(x)[:2].upper())
    df_avg = df.copy()
    df_avg['lang_id'] = 'AVERAGE'
    df_plot = pd.concat([df, df_avg], ignore_index=True)
    categories = sorted(df['lang_id'].unique().tolist()) + ['AVERAGE']
    df_plot['lang_id'] = pd.Categorical(df_plot['lang_id'], categories=categories, ordered=True)

    # --- 4. 辅助函数：边缘颜色微调 ---
    def adjust_lightness(color, amount=0.85):
        try:
            rgb = color[:3] if len(color) > 3 else color
            c = rgb_to_hsv(rgb)
            c[2] = max(0, min(1, c[2] * amount)) 
            return hsv_to_rgb(c)
        except:
            return color

    # --- 5. 绘图 ---
    fig, axes = joypy.joyplot(
        data=df_plot, column='norm_score', by='lang_id',
        figsize=(10, 6), overlap=2.5, x_range=[-5, 105],
        colormap=cm.Spectral_r, alpha=0.8, linewidth=1.2, 
        fade=True, bw_method=0.15,
        grid=False  
    )

    # --- 6. 核心修改：精细化包络线颜色 ---
    for ax in axes:
        lines = ax.get_lines()
        polys = ax.findobj(plt.matplotlib.collections.PolyCollection)
        if polys and lines:
            fill_color = polys[0].get_facecolor()[0]
            refined_color = adjust_lightness(fill_color, amount=0.85) 
            for line in lines:
                line.set_color(refined_color)
                line.set_alpha(0.9)

    # --- 7. 文本与标题覆盖 (y=0.94) ---
    target_font = 'DejaVu Sans'
    fig.suptitle('Preference Distribution by Language', 
                 fontsize=18, fontweight='normal', y=0.94, fontname=target_font)

    for i, ax in enumerate(axes):
        y_labels = ax.get_yticklabels()
        if y_labels:
            plt.setp(y_labels, fontname=target_font, fontweight='normal', fontsize=14)
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)

    # 底部 X 轴标签 (使用显示名 model_A/B)
    last_axis = axes[-1]
    tick_positions = [0, 25, 50, 75, 100]
    tick_labels = [
        f"Strongly prefer A\n({model_A})", "Weakly\nprefer A", 
        "No\npreference", "Weakly\nprefer B", f"Strongly prefer B\n({model_B})"
    ]
    last_axis.set_xticks(tick_positions)
    last_axis.set_xticklabels(tick_labels, fontsize=12, fontname=target_font, fontweight='normal')

    # 强制全图文本不加粗
    for text in fig.findobj(plt.Text):
        text.set_fontname(target_font)
        text.set_fontweight('normal')

    # --- 8. 超高清 PDF 导出 ---
    output_path = './final_academic_plot_clean.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=1200, transparent=True)
    plt.show()
    print(f"成功保存高清PDF至: {output_path}")
