import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

# --- Configuration & Paths ---
base_dir = '/Users/emma/Documents/linguistic_data/anew'
my_dataset_path = os.path.join(base_dir, 'final_words_clean - Sheet1.csv')
anew_path = os.path.join(base_dir, 'ANEW_words - Sheet1.csv')
output_folder = os.path.join(base_dir, 'valence_arousal_maps')

os.makedirs(output_folder, exist_ok=True)

# --- CLUTTER CONTROL SWITCH ---
# Set to False here so all 40 matched words get their text label appended
LABEL_ONLY_ANCHORS = False

# Example Anchor Words
ANCHOR_WORDS_LIST = ['Happy', 'Sad', 'Angry', 'Calm', 'Tired', 'Surprised', 'Neutral', 'Fearful', 'Disgusted', 'Bored']
ANCHOR_WORDS_LOWER = [w.strip().lower() for w in ANCHOR_WORDS_LIST]

# --- Word Lemma Mapping ---
LEMMA_MAP = {
    'comfort': 'comfortable',
    'excitement': 'excited',
    'thrill': 'thrilled',
    'shamed': 'ashamed',
    'ecstasy': 'ecstatic',
    'interest': 'interested',
    'horror': 'horrified',
    'sleep': 'sleepy',
    'annoy': 'annoyed',
    'misery': 'miserable',
    'cheer': 'cheerful',
    'delight': 'delighted',
    'concentrate': 'concentrating'
}

# --- Global Color Palette Configuration ---
COLOR_AROUSAL = '#FFB300'  
COLOR_VALENCE = '#8A2BE2'  

# Series 1: My Dataset
COLOR_MY_NORMAL = '#333333'  
COLOR_MY_ANCHOR = 'red'      
COLOR_MY_ERR = '#888888'  # Darkened for better visibility   

# Series 2: ANEW Dataset
COLOR_ANEW_NORMAL = '#00A896' 
COLOR_ANEW_ANCHOR = '#023E8A' 

# --- Core Axis Engine ---
def setup_va_axes(ax, title):
    ax.set_title(title, fontsize=38, fontweight='bold', pad=35, color='#444444')
    
    # Strictly locked back to 1.1 for breathing room, ticks strictly locked to 1.0 max.
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal', adjustable='box') 
    
    ax.set_xticks(np.arange(-1.0, 1.1, 0.2)) 
    ax.set_yticks(np.arange(-1.0, 1.1, 0.2))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    
    ax.tick_params(axis='x', direction='inout', labelsize=13, colors='#666666')
    ax.tick_params(axis='y', direction='inout', labelsize=13, colors='#666666')

# --- Data Loading & Normalization ---
def load_datasets():
    print("Loading datasets...")
    df_my = pd.read_csv(my_dataset_path)
    df_my.columns = df_my.columns.str.strip().str.lower()
    df_my['word_clean'] = df_my['word'].astype(str).str.strip()
    df_my['word_lower'] = df_my['word_clean'].str.lower()
    
    df_anew = pd.read_csv(anew_path)
    df_anew.columns = df_anew.columns.str.strip().str.lower()
    df_anew = df_anew.dropna(subset=['valence_mean', 'arousal_mean'])
    
    df_anew['word_clean'] = df_anew['word'].astype(str).str.replace('*', '', regex=False).str.strip()
    df_anew['word_lower'] = df_anew['word_clean'].str.lower()
    
    df_anew['word_mapped'] = df_anew['word_lower'].apply(lambda w: LEMMA_MAP.get(w, w))
    
    df_anew['valence_norm'] = (df_anew['valence_mean'] - 1) / 4 - 1
    df_anew['arousal_norm'] = (df_anew['arousal_mean'] - 1) / 4 - 1
    
    return df_my, df_anew

# --- Execution Pipeline ---
def generate_unified_comparison_map():
    df_my, df_anew = load_datasets()
    
    my_lookup = df_my.set_index('word_lower')[['valence_mean', 'arousal_mean']].to_dict('index')
    valid_words = set(my_lookup.keys())
    
    fig, ax = plt.subplots(figsize=(32, 32), dpi=300)
    setup_va_axes(ax, "New VA-Word Mappings vs. Mappings of ANEW Valence-Arousal")
    
    # Restored to exact quadrant positions
    ax.annotate('Valence', (0.45, 0.05), xycoords='data', color=COLOR_VALENCE, fontsize=36, fontweight='bold', zorder=10)
    ax.annotate('Arousal', (0.05, 0.55), xycoords='data', color=COLOR_AROUSAL, fontsize=36, fontweight='bold', zorder=10)
    
    texts = []
    
    # Helper function to draw tether lines with a midpoint arrow
    def draw_tether_with_arrow(anew_x, anew_y, my_x, my_y, color):
        dx = my_x - anew_x
        dy = my_y - anew_y
        mid_x = anew_x + (dx / 2)
        mid_y = anew_y + (dy / 2)
        
        # Main bolder tether line
        ax.plot([my_x, anew_x], [my_y, anew_y], 
                color=color, linestyle='-', linewidth=2.5, alpha=0.85, zorder=1)
        
        # Midpoint directional arrow
        if dx != 0 or dy != 0:
            ax.annotate('', 
                        xy=(mid_x, mid_y), 
                        xytext=(mid_x - dx*0.05, mid_y - dy*0.05),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5, alpha=0.85, mutation_scale=25),
                        zorder=2)

    print("Plotting 'My Dataset' Layer...")
    for _, row in df_my.iterrows():
        is_anchor = row['word_lower'] in ANCHOR_WORDS_LOWER
        pt_color = COLOR_MY_ANCHOR if is_anchor else COLOR_MY_NORMAL
        
        # Thicker error bars to match LLM chart
        if 'valence_se' in row and 'arousal_se' in row and pd.notnull(row['valence_se']):
            ax.errorbar(row['valence_mean'], row['arousal_mean'], 
                        xerr=row['valence_se'], yerr=row['arousal_se'],
                        fmt='o', markersize=0, ecolor=COLOR_MY_ANCHOR if is_anchor else COLOR_MY_ERR, 
                        elinewidth=1.5, capsize=3, alpha=0.85, zorder=3)
                        
        ax.scatter(row['valence_mean'], row['arousal_mean'], color=pt_color, s=65, edgecolors='none', zorder=7)
        
        # Append text ONLY for the new human dataset mapping
        if not LABEL_ONLY_ANCHORS or (LABEL_ONLY_ANCHORS and is_anchor):
            texts.append(ax.text(row['valence_mean'], row['arousal_mean'], row['word_clean'], 
                                 fontsize=26, color=pt_color, fontweight=('bold'), zorder=9))

    print("Plotting Normalized ANEW Layer & Tether Vectors...")
    df_anew_plot = df_anew[df_anew['word_mapped'].isin(valid_words)]
    
    for _, row in df_anew_plot.iterrows():
        w_mapped = row['word_mapped']
        is_anchor = w_mapped in ANCHOR_WORDS_LOWER 
        pt_color = COLOR_ANEW_ANCHOR if is_anchor else COLOR_ANEW_NORMAL
        
        my_coords = my_lookup[w_mapped]
        
        # Draw the tethers using the arrow helper function
        draw_tether_with_arrow(row['valence_norm'], row['arousal_norm'], my_coords['valence_mean'], my_coords['arousal_mean'], pt_color)
        
        ax.scatter(row['valence_norm'], row['arousal_norm'], color=pt_color, s=65, marker='^', zorder=5)
        
        # APPEND TEXT FOR ANEW LAYER (Italicized and slightly smaller for visual hierarchy)
        if not LABEL_ONLY_ANCHORS or (LABEL_ONLY_ANCHORS and is_anchor):
            texts.append(ax.text(row['valence_norm'], row['arousal_norm'], row['word_clean'], 
                                 fontsize=22, color=pt_color, fontstyle='italic', fontweight=('bold' if is_anchor else 'normal'), zorder=8))

    # Flattened legend (4 columns)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_MY_NORMAL, markersize=14, label='New VA-Word Mappings (Standard)'),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_MY_ANCHOR, markersize=14, label='New VA-Word Mappings (Anchor)'),
        plt.Line2D([0], [0], marker='^', color='none', markerfacecolor=COLOR_ANEW_NORMAL, markersize=14, label='ANEW Normalized Mappings (Standard)'),
        plt.Line2D([0], [0], marker='^', color='none', markerfacecolor=COLOR_ANEW_ANCHOR, markersize=14, label='ANEW Normalized Mappings (Anchor)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03), 
              ncol=4, frameon=True, facecolor='#ffffff', edgecolor='#e0e0e0', fontsize=22)

    print(f"Executing mathematical overlap adjustments for {len(texts)} series string elements...")
    
    # Adjusted repulsion parameters to mirror the LLM script
    adjust_text(texts, 
                force_points=0.4, 
                force_text=1.2,
                expand_text=(1.4, 1.6), 
                lim=5000, 
                only_move={'text':'xy'},
                arrowprops=dict(arrowstyle='-', color='#a0a0a0', alpha=0.8, lw=0.8), 
                zorder=2)

    # --- THE SQUARE FIX ---
    # Manually pad the internal layout so the title/legend fit inside the 32x32 dimensions.
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.92)

    save_path = os.path.join(output_folder, 'my_dataset_vs_anew_comparison.png')
    print(f"Exporting comprehensive canvas plot to file grid...")
    
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Successfully generated map output at: {save_path}")

if __name__ == "__main__":
    generate_unified_comparison_map()