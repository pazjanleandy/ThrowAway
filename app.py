import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, messagebox, filedialog
import threading
import os
import glob
from PIL import Image, ImageTk
import testing
import cv2
import numpy as np
import queue
import time
from datetime import datetime
import json
from webcam_utils import correct_fisheye, adjust_image

# --- Modern Color Palette ---
BG_DARK = "#1A1F2E"          # Deep navy background
BG_CARD = "#242B3D"          # Card background
SIDEBAR_BG = "#1E2532"       # Sidebar background
ACCENT_GREEN = "#10B981"     # Primary green accent
ACCENT_BLUE = "#3B82F6"      # Blue accent
ACCENT_PURPLE = "#8B5CF6"    # Purple accent
ACCENT_ORANGE = "#F59E0B"    # Orange accent
WHITE = "#F8FAFC"            # Clean white
GRAY_LIGHT = "#94A3B8"       # Light gray text
GRAY_MID = "#64748B"         # Medium gray
GRAY_DARK = "#475569"        # Dark gray
SUCCESS = "#10B981"          # Success green
WARNING = "#F59E0B"          # Warning orange
ERROR = "#EF4444"            # Error red
HOVER_BG = "#2D3748"         # Hover background

FONT_TITLE = ("Segoe UI", 24, "bold")
FONT_SUBTITLE = ("Segoe UI", 18, "bold") 
FONT_BTN = ("Segoe UI", 11, "bold")
FONT_LABEL = ("Segoe UI", 10, "bold")
FONT_TEXT = ("Segoe UI", 9)
FONT_OUTPUT = ("Consolas", 10)
FONT_ICON = ("Segoe UI Emoji", 16)
FONT_ICON_LARGE = ("Segoe UI Emoji", 20)

# --- Icon Configuration ---
ICONS = {
    'home': '🏠',
    'train': '🎯',
    'predict': '📷',
    'retrain': '🔄',
    'database': '📊',
    'start': '▶️',
    'stop': '⏹️',
    'settings': '⚙️',
    'back': '◀️',
    'forward': '▶️',
    'refresh': '🔄',
    'save': '💾',
    'load': '📁',
    'delete': '🗑️',
    'edit': '✏️',
    'view': '👁️',
    'chart': '📈',
    'image': '🖼️',
    'camera': '📹',
    'model': '🤖',
    'data': '📋',
    'analytics': '📊',
    'success': '✅',
    'warning': '⚠️',
    'error': '❌',
    'info': 'ℹ️',
    'config': '🔧',
    'upload': '📤',
    'download': '📥',
    'plus': '➕',
    'minus': '➖',
    'play': '▶️',
    'pause': '⏸️',
    'reset': '🔄',
    'folder': '📂',
    'file': '📄',
    'search': '🔍',
    'filter': '🔽',
    'sort': '📶',
    'menu': '☰',
    'close': '✖️',
    'maximize': '🔳',
    'minimize': '🔲'
}

# --- Paths ---
database_folder = testing.database_folder

# --- Thread Helper ---
def run_in_thread(target):
    def wrapper(*args, **kwargs):
        threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True).start()
    return wrapper

# --- Main App ---
class ThrowawayApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Throwaway")
        self.configure(bg=BG_DARK)
        self.state('zoomed')  # Start maximized
        self.minsize(1200, 700)
        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.current_frame = None
        self.show_home()

    def clear_frame(self):
        if self.current_frame:
            self.current_frame.destroy()
            self.current_frame = None

    def show_home(self):
        self.clear_frame()
        self.current_frame = HomePage(self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_main_menu(self):
        self.clear_frame()
        self.current_frame = MainMenu(self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_dashboard(self):
        self.clear_frame()
        self.current_frame = Dashboard(self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_database(self):
        self.clear_frame()
        self.current_frame = DatabasePage(self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def show_retrain_menu(self):
        self.clear_frame()
        self.current_frame = RetrainMenu(self)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def on_exit(self):
        self.destroy()

# --- Modern Sidebar Button ---
class SidebarButton(tk.Frame):
    def __init__(self, parent, text, command, selected=False, icon=None):
        super().__init__(parent, bg=SIDEBAR_BG, highlightthickness=0)
        self.selected = selected
        self.text = text
        self.command = command
        self.icon = icon
        
        # Create button with modern styling and icon
        display_text = f"{icon} {text}" if icon else text
        self.btn = tk.Button(
            self, text=display_text, font=FONT_BTN, 
            fg=WHITE if selected else GRAY_LIGHT, 
            bg=ACCENT_BLUE if selected else SIDEBAR_BG,
            activebackground=HOVER_BG, 
            activeforeground=WHITE, 
            bd=0, relief=tk.FLAT, 
            command=command, cursor="hand2",
            padx=20, pady=12, anchor="w"
        )
        self.btn.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        # Add hover effects
        self.btn.bind("<Enter>", self.on_enter)
        self.btn.bind("<Leave>", self.on_leave)
    
    def on_enter(self, event):
        if not self.selected:
            self.btn.config(bg=HOVER_BG, fg=WHITE)
    
    def on_leave(self, event):
        if not self.selected:
            self.btn.config(bg=SIDEBAR_BG, fg=GRAY_LIGHT)

# --- Modern Home Page ---
class HomePage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_DARK)
        self.master = master
        
        # Main container with padding
        main_container = tk.Frame(self, bg=BG_DARK)
        main_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Header section
        header_frame = tk.Frame(main_container, bg=BG_DARK)
        header_frame.pack(fill=tk.X, pady=(0, 40))
        
        # Title with modern styling
        title = tk.Label(header_frame, text="THROWAWAY", font=FONT_TITLE, fg=ACCENT_GREEN, bg=BG_DARK)
        title.pack(pady=(20, 5))
        subtitle = tk.Label(header_frame, text="AI-Powered Trash Classification System", 
                           font=FONT_SUBTITLE, fg=GRAY_LIGHT, bg=BG_DARK)
        subtitle.pack(pady=(0, 5))
        description = tk.Label(header_frame, text="Intelligent waste sorting using computer vision and machine learning", 
                              font=FONT_TEXT, fg=GRAY_MID, bg=BG_DARK)
        description.pack()
        
        # Cards container
        cards_frame = tk.Frame(main_container, bg=BG_DARK)
        cards_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 30))
        
        # Create modern action cards with icons
        buttons = [
            ("Train Model", "Start training a new classification model", self.master.show_main_menu, ACCENT_BLUE, ICONS['train']),
            ("Live Classification", "Real-time waste classification dashboard", self.master.show_dashboard, ACCENT_GREEN, ICONS['predict']),
            ("View Database", "Browse classification history and results", self.master.show_database, ACCENT_PURPLE, ICONS['database']),
            ("Retrain Model", "Improve model with new feedback data", self.master.show_retrain_menu, ACCENT_ORANGE, ICONS['retrain']),
            ("System Maintenance", "Clean up non-image files from dataset", run_in_thread(self.remove_files), WARNING, ICONS['config']),
            ("Exit Application", "Close the application safely", self.master.on_exit, ERROR, ICONS['close'])
        ]
        
        for i, (title_text, desc_text, cmd, accent, icon) in enumerate(buttons):
            card = self.create_action_card(cards_frame, title_text, desc_text, cmd, accent, icon)
            card.grid(row=i//2, column=i%2, padx=15, pady=15, sticky="ew")
        
        # Configure grid weights for responsive layout
        cards_frame.grid_columnconfigure(0, weight=1)
        cards_frame.grid_columnconfigure(1, weight=1)
        
        # Status panel with modern styling
        status_frame = tk.Frame(main_container, bg=BG_CARD, height=60)
        status_frame.pack(fill=tk.X, pady=(0, 0))
        status_frame.pack_propagate(False)
        
        status_border = tk.Frame(status_frame, bg=SUCCESS, height=3)
        status_border.pack(fill=tk.X)
        
        status_content = tk.Frame(status_frame, bg=BG_CARD)
        status_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        tk.Label(status_content, text="System Status:", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(side=tk.LEFT)
        self.status = tk.StringVar(value="READY")
        status_label = tk.Label(status_content, textvariable=self.status, font=FONT_BTN, fg=SUCCESS, bg=BG_CARD)
        status_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def create_action_card(self, parent, title_text, desc_text, command, accent_color, icon=None):
        # Main card frame
        card = tk.Frame(parent, bg=BG_CARD, height=100)
        card.pack_propagate(False)
        
        # Accent border
        accent_border = tk.Frame(card, bg=accent_color, height=4)
        accent_border.pack(fill=tk.X)
        
        # Card content
        content = tk.Frame(card, bg=BG_CARD)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # Icon and title layout
        header_frame = tk.Frame(content, bg=BG_CARD)
        header_frame.pack(fill=tk.X, anchor=tk.W)
        
        if icon:
            # Icon label
            icon_label = tk.Label(header_frame, text=icon, font=FONT_ICON_LARGE, fg=accent_color, bg=BG_CARD)
            icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Title
        title_label = tk.Label(header_frame, text=title_text, font=FONT_BTN, fg=WHITE, bg=BG_CARD)
        title_label.pack(side=tk.LEFT, anchor=tk.W)
        
        # Description
        desc_label = tk.Label(content, text=desc_text, font=FONT_TEXT, fg=GRAY_LIGHT, bg=BG_CARD, wraplength=250)
        desc_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Bind click events
        components = [card, accent_border, content, header_frame, title_label, desc_label]
        if icon:
            components.append(icon_label)
        for comp in components:
            comp.bind("<Button-1>", lambda e: command() if callable(command) else None)
            comp.bind("<Enter>", lambda e: self.card_hover_enter(card, content, header_frame, title_label, desc_label, accent_color, icon_label if icon else None))
            comp.bind("<Leave>", lambda e: self.card_hover_leave(card, content, header_frame, title_label, desc_label, icon_label if icon else None))
            comp.configure(cursor="hand2")
        
        return card
    
    def card_hover_enter(self, card, content, header_frame, title_label, desc_label, accent_color, icon_label=None):
        card.config(bg=HOVER_BG)
        content.config(bg=HOVER_BG)
        header_frame.config(bg=HOVER_BG)
        title_label.config(bg=HOVER_BG, fg=accent_color)
        desc_label.config(bg=HOVER_BG, fg=WHITE)
        if icon_label:
            icon_label.config(bg=HOVER_BG)
    
    def card_hover_leave(self, card, content, header_frame, title_label, desc_label, icon_label=None):
        card.config(bg=BG_CARD)
        content.config(bg=BG_CARD)
        header_frame.config(bg=BG_CARD)
        title_label.config(bg=BG_CARD, fg=WHITE)
        desc_label.config(bg=BG_CARD, fg=GRAY_LIGHT)
        if icon_label:
            icon_label.config(bg=BG_CARD)

    def retrain_model(self):
        self.status.set("Retraining model...")
        def run():
            try:
                testing.retrain_model()  # No callbacks, just call retrain_model
                self.status.set("Retraining complete.")
            except Exception as e:
                self.status.set(f"Error: {e}")
        threading.Thread(target=run, daemon=True).start()

    def remove_files(self):
        self.status.set("Removing non-image files...")
        try:
            testing.remove_non_image_files(testing.original_dataset_path)
            testing.remove_non_image_files(testing.augmented_dataset_path)
            testing.remove_non_image_files(testing.feedback_dataset_path)
            testing.remove_non_image_files(testing.combined_dataset_path)
            self.status.set("Non-image files removed.")
        except Exception as e:
            self.status.set(f"Error: {e}")

# --- Main Menu (for training) ---
class MainMenu(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_DARK)
        self.master = master
        self.log_queue = queue.Queue()
        self.dataset_mode = tk.StringVar(value="augmented")
        self.model_type = tk.StringVar(value="cnn")  # 'cnn' or 'transfer'
        self.epochs_var = tk.IntVar(value=10)  # Default to 10 epochs
        self.create_layout()
        self.training_log_updater()

    def create_layout(self):
        sidebar = tk.Frame(self, bg=SIDEBAR_BG, width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Sidebar header
        tk.Label(sidebar, text="TRAINING", font=FONT_TITLE, fg=ACCENT_BLUE, bg=SIDEBAR_BG).pack(pady=(30, 20))
        tk.Label(sidebar, text="Model Training Center", font=FONT_TEXT, fg=GRAY_LIGHT, bg=SIDEBAR_BG).pack(pady=(0, 30))
        
        SidebarButton(sidebar, "Back to Home", self.master.show_home, selected=True, icon=ICONS['home']).pack(fill=tk.X, pady=8, padx=18)
        
        # Status section
        status_section = tk.Frame(sidebar, bg=BG_CARD)
        status_section.pack(side=tk.BOTTOM, fill=tk.X, padx=18, pady=20)
        
        tk.Frame(status_section, bg=ACCENT_BLUE, height=3).pack(fill=tk.X)
        status_content = tk.Frame(status_section, bg=BG_CARD)
        status_content.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(status_content, text="Status:", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W)
        self.status = tk.StringVar(value="Ready to train")
        tk.Label(status_content, textvariable=self.status, fg=WHITE, bg=BG_CARD, font=FONT_TEXT, wraplength=220).pack(anchor=tk.W, pady=(5, 0))

        main = tk.Frame(self, bg=BG_DARK)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Training configuration card
        config_card = tk.Frame(main, bg=BG_CARD)
        config_card.pack(fill=tk.X, pady=(0, 20))
        
        tk.Frame(config_card, bg=ACCENT_BLUE, height=3).pack(fill=tk.X)
        config_content = tk.Frame(config_card, bg=BG_CARD)
        config_content.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(config_content, text="Training Configuration", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 15))
        
        # Choose model run subfolder root
        runroot_frame = tk.Frame(config_content, bg=BG_CARD)
        runroot_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(runroot_frame, text="Model Save Root", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 8))
        self.runroot_var = tk.StringVar(value=getattr(testing, 'models_root', os.path.join(os.getcwd(), 'models')))
        tk.Entry(runroot_frame, textvariable=self.runroot_var, width=50, font=FONT_TEXT, bg=SIDEBAR_BG, fg=WHITE, insertbackground=WHITE).pack(anchor=tk.W, padx=10)

        # Training log card
        log_card = tk.Frame(main, bg=BG_CARD)
        log_card.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        tk.Frame(log_card, bg=ACCENT_GREEN, height=3).pack(fill=tk.X)
        log_header = tk.Frame(log_card, bg=BG_CARD)
        log_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        
        tk.Label(log_header, text="Training Progress", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(log_card, height=15, font=FONT_OUTPUT, bg=SIDEBAR_BG, fg=WHITE, 
                                                 state=tk.DISABLED, borderwidth=0, highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))

        # Move configuration controls into the config card
        # Model type selection
        model_frame = tk.Frame(config_content, bg=BG_CARD)
        model_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(model_frame, text="Model Type", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 8))
        model_options = [
            ("Standard CNN", "cnn"),
            ("Transfer Learning (MobileNetV2)", "transfer"),
        ]
        for text, value in model_options:
            tk.Radiobutton(model_frame, text=text, variable=self.model_type, value=value, 
                          font=FONT_TEXT, fg=WHITE, bg=BG_CARD, selectcolor=ACCENT_BLUE, 
                          activebackground=BG_CARD, activeforeground=ACCENT_BLUE).pack(anchor=tk.W, padx=10)
        
        # Dataset selection
        dataset_frame = tk.Frame(config_content, bg=BG_CARD)
        dataset_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(dataset_frame, text="Dataset Mode", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 8))
        dataset_options = [
            ("Original Only", "original"),
            ("Augmented Only", "augmented"),
            ("Both (Original + Augmented)", "both"),
        ]
        for text, value in dataset_options:
            tk.Radiobutton(dataset_frame, text=text, variable=self.dataset_mode, value=value, 
                          font=FONT_TEXT, fg=WHITE, bg=BG_CARD, selectcolor=ACCENT_BLUE, 
                          activebackground=BG_CARD, activeforeground=ACCENT_BLUE).pack(anchor=tk.W, padx=10)
        
        # Epoch selection
        epochs_frame = tk.Frame(config_content, bg=BG_CARD)
        epochs_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(epochs_frame, text="Number of Epochs", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 8))
        epochs_input_frame = tk.Frame(epochs_frame, bg=BG_CARD)
        epochs_input_frame.pack(anchor=tk.W, padx=10)
        tk.Spinbox(epochs_input_frame, from_=1, to=1000, textvariable=self.epochs_var, width=8, 
                  font=FONT_TEXT, bg=SIDEBAR_BG, fg=WHITE, insertbackground=WHITE).pack(side=tk.LEFT)
        
        # Start training button
        btn_frame = tk.Frame(config_content, bg=BG_CARD)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        tk.Button(btn_frame, text=f"{ICONS['start']} Start Training", font=FONT_BTN, fg=WHITE, bg=ACCENT_BLUE, 
                 activebackground=ACCENT_GREEN, activeforeground=WHITE, bd=0, relief=tk.FLAT, 
                 width=15, height=2, cursor="hand2", command=run_in_thread(self.train_model)).pack(anchor=tk.W)
        # Progress tracking card
        progress_card = tk.Frame(main, bg=BG_CARD)
        progress_card.pack(fill=tk.X, pady=(0, 20))
        
        tk.Frame(progress_card, bg=ACCENT_ORANGE, height=3).pack(fill=tk.X)
        progress_content = tk.Frame(progress_card, bg=BG_CARD)
        progress_content.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(progress_content, text="Training Progress", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 15))
        
        # Preparation progress
        self.prep_progress_var = tk.DoubleVar()
        self.prep_label = tk.Label(progress_content, text="Data Preparation", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD)
        self.prep_label.pack(anchor=tk.W, pady=(0, 5))
        self.prep_progress_bar = ttk.Progressbar(progress_content, variable=self.prep_progress_var, maximum=1, length=500, mode='determinate')
        self.prep_progress_bar.pack(anchor=tk.W, pady=(0, 15))
        
        # Epoch progress
        self.epoch_label = tk.Label(progress_content, text="Epoch Progress", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD)
        self.epoch_label.pack(anchor=tk.W, pady=(0, 5))
        self.current_epoch_var = tk.StringVar(value="Current Epoch: -/-")
        self.current_epoch_label = tk.Label(progress_content, textvariable=self.current_epoch_var, font=FONT_TEXT, fg=ACCENT_BLUE, bg=BG_CARD)
        self.current_epoch_label.pack(anchor=tk.W)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_content, variable=self.progress_var, maximum=10, length=500)
        self.progress_bar.pack(anchor=tk.W, pady=(5, 15))
        
        # Batch progress
        self.batch_label = tk.Label(progress_content, text="Batch Progress", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD)
        self.batch_label.pack(anchor=tk.W, pady=(0, 5))
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress_bar = ttk.Progressbar(progress_content, variable=self.batch_progress_var, maximum=1, length=500)
        self.batch_progress_bar.pack(anchor=tk.W)
        
        # Results visualization card
        results_card = tk.Frame(main, bg=BG_CARD)
        results_card.pack(fill=tk.BOTH, expand=True)
        
        tk.Frame(results_card, bg=ACCENT_PURPLE, height=3).pack(fill=tk.X)
        results_header = tk.Frame(results_card, bg=BG_CARD)
        results_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        
        tk.Label(results_header, text="Training Results", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        
        # Panel for plots
        self.plot_panel = tk.Frame(results_card, bg=BG_CARD)
        self.plot_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.acc_img_label = tk.Label(self.plot_panel, bg=BG_CARD)
        self.acc_img_label.pack(side=tk.LEFT, padx=(0, 10))
        self.loss_img_label = tk.Label(self.plot_panel, bg=BG_CARD)
        self.loss_img_label.pack(side=tk.LEFT, padx=(10, 0))

    def set_status(self, msg):
        self.status.set(msg)

    def training_log_updater(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(200, self.training_log_updater)

    def train_model(self):
        self.set_status("Preparing to start training...")
        self.progress_var.set(0)
        self.batch_progress_var.set(0)
        self.prep_progress_var.set(0)
        self.current_epoch_var.set("Current Epoch: -/-")
        self.prep_progress_bar['maximum'] = len(testing.categories) if hasattr(testing, 'categories') else 1
        self.prep_progress_bar['value'] = 0
        self.prep_progress_bar.lift()  # Show preparation bar on top
        self.prep_label.lift()
        self.progress_bar.lower()      # Hide epoch bar initially
        self.epoch_label.lower()
        self.current_epoch_label.lower()
        self.batch_progress_bar.lower()# Hide batch bar initially
        self.batch_label.lower()
        def run():
            def progress_callback(kind, current, total=None):
                if kind == 'preparing':
                    self.prep_progress_bar['maximum'] = total
                    self.prep_progress_var.set(current)
                    self.set_status(f"Preparing data: {current}/{total}...")
                elif kind == 'epoch':
                    self.progress_var.set(current)
                    self.current_epoch_var.set(f"Current Epoch: {current}/{total}")
                elif kind == 'batch':
                    self.batch_progress_bar['maximum'] = total
                    self.batch_progress_var.set(current)
                elif kind == 'status':
                    if current == 'preparing':
                        self.set_status("Preparing for training...")
                    elif current == 'training':
                        self.set_status("Training in progress...")
                    elif current == 'done':
                        self.set_status("Training complete.")
            try:
                # Call PyTorch training script externally
                testing.train_model(
                    model_type=self.model_type.get(),
                    dataset_mode=self.dataset_mode.get(),
                    epochs=self.epochs_var.get(), # Fixed epochs for training
                    log_queue=self.log_queue,
                    progress_callback=progress_callback
                )
                self.set_status("Training complete.")
                self.show_training_plots()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.set_status(f"Error: {e}")
        threading.Thread(target=run, daemon=True).start()

    def show_training_plots(self):
        import os
        from PIL import Image, ImageTk
        acc_path = os.path.join(testing.database_folder, 'training_plot.png')
        loss_path = os.path.join(testing.database_folder, 'loss_plot.png')
        if os.path.exists(acc_path):
            acc_img = Image.open(acc_path)
            acc_img = acc_img.resize((320, 240))
            acc_imgtk = ImageTk.PhotoImage(acc_img)
            self.acc_img_label.imgtk = acc_imgtk
            self.acc_img_label.config(image=acc_imgtk, text="")
        else:
            self.acc_img_label.config(image="", text="")
        if os.path.exists(loss_path):
            loss_img = Image.open(loss_path)
            loss_img = loss_img.resize((320, 240))
            loss_imgtk = ImageTk.PhotoImage(loss_img)
            self.loss_img_label.imgtk = loss_imgtk
            self.loss_img_label.config(image=loss_imgtk, text="")
        else:
            self.loss_img_label.config(image="", text="")

# --- Dashboard (Predict) ---
class Dashboard(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_DARK)
        self.master = master
        self.running = False
        self.stop_event = threading.Event()
        self.prediction_text = tk.StringVar()
        self.category_counts = {cat: 0 for cat in testing.categories}
        self.current_roi = None
        self.current_label = None
        self.current_confidence = None
        # Detection logging (triggered serial sends)
        self.detection_records = []  # list of dicts: {type, confidence, timestamp}
        self.detection_counts = {cat: 0 for cat in testing.categories}
        self.detection_conf_sums = {cat: 0.0 for cat in testing.categories}
        # Fisheye and adjustment states
        self.fisheye_enabled = tk.BooleanVar(value=False)
        self.brightness = tk.DoubleVar(value=1.0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.saturation = tk.DoubleVar(value=1.0)
        # Model selection state
        self.cnn_models_dir = tk.StringVar(value=os.path.join(os.getcwd(), 'model'))
        self.yolo_models_dir = tk.StringVar(value=os.path.join(os.getcwd(), 'Waste-Classification-using-YOLOv8', 'streamlit-detection-tracking - app', 'weights'))
        self.cnn_model_var = tk.StringVar(value='')
        self.yolo_model_var = tk.StringVar(value='')
        # UI elements to be created later
        self.cnn_dropdown = None
        self.yolo_dropdown = None
        self.log_text = None
        self.scrollbar_shown = False
        self.create_layout()
        self.pred_thread = None
        self._export_done = False

    def create_layout(self):
        # Scrollable container for entire dashboard content
        canvas = tk.Canvas(self, bg=BG_DARK, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=v_scrollbar.set)
        # Only pack scrollbar when prediction starts
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content = tk.Frame(canvas, bg=BG_DARK)
        content_id = canvas.create_window((0, 0), window=content, anchor="nw")
        self.canvas = canvas
        self.v_scrollbar = v_scrollbar

        def _on_content_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        content.bind("<Configure>", _on_content_configure)

        def _on_canvas_configure(event):
            # Make inner frame width match the canvas width
            canvas.itemconfig(content_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Enable mouse wheel scrolling on Windows
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        content.bind_all("<MouseWheel>", _on_mousewheel)

        sidebar = tk.Frame(content, bg=SIDEBAR_BG, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Sidebar header
        tk.Label(sidebar, text="DASHBOARD", font=FONT_TITLE, fg=ACCENT_GREEN, bg=SIDEBAR_BG).pack(pady=(30, 20))
        tk.Label(sidebar, text="Live Classification", font=FONT_TEXT, fg=GRAY_LIGHT, bg=SIDEBAR_BG).pack(pady=(0, 30))
        
        SidebarButton(sidebar, "Back to Home", self.stop_and_back, selected=True, icon=ICONS['home']).pack(fill=tk.X, pady=8, padx=18)
        # Camera settings card
        settings_card = tk.Frame(sidebar, bg=BG_CARD)
        settings_card.pack(fill=tk.X, padx=18, pady=(0, 20))
        
        tk.Frame(settings_card, bg=ACCENT_GREEN, height=3).pack(fill=tk.X)
        settings_content = tk.Frame(settings_card, bg=BG_CARD)
        settings_content.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(settings_content, text="Camera Settings", font=FONT_LABEL, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 15))
        
        # Fisheye toggle
        fisheye_btn = tk.Checkbutton(settings_content, text="Fisheye Correction", variable=self.fisheye_enabled, 
                                   font=FONT_TEXT, fg=WHITE, bg=BG_CARD, selectcolor=ACCENT_GREEN, 
                                   activebackground=BG_CARD, activeforeground=ACCENT_GREEN)
        fisheye_btn.pack(anchor="w", pady=(0, 15))
        
        # Brightness slider
        tk.Label(settings_content, text="Brightness", font=FONT_TEXT, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor="w", pady=(0, 5))
        tk.Scale(settings_content, from_=0.5, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.brightness, 
                length=200, bg=BG_CARD, fg=WHITE, troughcolor=ACCENT_GREEN, highlightthickness=0, 
                activebackground=ACCENT_GREEN).pack(anchor="w", pady=(0, 10))
        
        # Contrast slider
        tk.Label(settings_content, text="Contrast", font=FONT_TEXT, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor="w", pady=(0, 5))
        tk.Scale(settings_content, from_=0.5, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.contrast, 
                length=200, bg=BG_CARD, fg=WHITE, troughcolor=ACCENT_GREEN, highlightthickness=0, 
                activebackground=ACCENT_GREEN).pack(anchor="w", pady=(0, 10))
        
        # Saturation slider
        tk.Label(settings_content, text="Saturation", font=FONT_TEXT, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor="w", pady=(0, 5))
        tk.Scale(settings_content, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.saturation, 
                length=200, bg=BG_CARD, fg=WHITE, troughcolor=ACCENT_GREEN, highlightthickness=0, 
                activebackground=ACCENT_GREEN).pack(anchor="w", pady=(0, 15))
        
        # Reset button
        tk.Button(settings_content, text="Reset Settings", command=self.reset_webcam_settings, 
                 bg=WARNING, fg=WHITE, font=FONT_BTN, bd=0, relief=tk.FLAT, 
                 activebackground=ACCENT_ORANGE, cursor="hand2").pack(fill=tk.X)

        main = tk.Frame(content, bg=BG_DARK)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Top section with webcam and counter
        top_panel = tk.Frame(main, bg=BG_DARK)
        top_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Webcam feed card
        webcam_card = tk.Frame(top_panel, bg=BG_CARD)
        webcam_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        tk.Frame(webcam_card, bg=ACCENT_BLUE, height=3).pack(fill=tk.X)
        webcam_header = tk.Frame(webcam_card, bg=BG_CARD)
        webcam_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        tk.Label(webcam_header, text="Live Camera Feed", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        
        self.webcam_label = tk.Label(webcam_card, bg=BG_CARD)
        self.webcam_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Category counter card
        counter_card = tk.Frame(top_panel, bg=BG_CARD)
        counter_card.pack(side=tk.LEFT, fill=tk.Y, padx=(15, 0))
        
        tk.Frame(counter_card, bg=ACCENT_PURPLE, height=3).pack(fill=tk.X)
        counter_header = tk.Frame(counter_card, bg=BG_CARD)
        counter_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        tk.Label(counter_header, text="Detection Counter", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        
        counter_content = tk.Frame(counter_card, bg=BG_CARD)
        counter_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.counter_labels = {}
        colors = [ACCENT_GREEN, ACCENT_BLUE, ACCENT_PURPLE, ACCENT_ORANGE, WARNING, ERROR]
        for i, cat in enumerate(testing.categories):
            frame = tk.Frame(counter_content, bg=BG_CARD)
            frame.pack(fill=tk.X, pady=8)
            
            # Color indicator
            indicator = tk.Frame(frame, bg=colors[i % len(colors)], width=4, height=20)
            indicator.pack(side=tk.LEFT, padx=(0, 10))
            
            lbl = tk.Label(frame, text=f"{cat.capitalize()}", font=FONT_TEXT, fg=WHITE, bg=BG_CARD, anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            val = tk.Label(frame, text="0", font=FONT_BTN, fg=colors[i % len(colors)], bg=BG_CARD, anchor="e")
            val.pack(side=tk.RIGHT)
            self.counter_labels[cat] = val
        # Helper to refresh UI counters from detection_counts
        def _make_update_func(self_ref):
            def _update():
                try:
                    for cat in testing.categories:
                        val = str(int(self_ref.detection_counts.get(cat, 0)))
                        self_ref.counter_labels[cat].config(text=val)
                except Exception:
                    pass
            return _update
        self._update_counters_ui = _make_update_func(self)
        # Bottom section with prediction output, mini log (left) and controls (right)
        bottom_panel = tk.Frame(main, bg=BG_DARK)
        bottom_panel.pack(fill=tk.BOTH, expand=True)

        left_column = tk.Frame(bottom_panel, bg=BG_DARK)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Prediction output card
        output_card = tk.Frame(left_column, bg=BG_CARD)
        output_card.pack(fill=tk.BOTH, expand=True, padx=(0, 15))
        
        tk.Frame(output_card, bg=SUCCESS, height=3).pack(fill=tk.X)
        output_header = tk.Frame(output_card, bg=BG_CARD)
        output_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        tk.Label(output_header, text="Prediction Results", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        
        output_content = tk.Frame(output_card, bg=BG_CARD)
        output_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.output_label = tk.Label(output_content, textvariable=self.prediction_text, font=FONT_TEXT, 
                                   fg=GRAY_LIGHT, bg=BG_CARD, anchor="nw", justify=tk.LEFT, wraplength=400)
        self.output_label.pack(fill=tk.BOTH, expand=True)
        
        # Mini log panel under prediction results (left column)
        log_card = tk.Frame(left_column, bg=BG_CARD)
        log_card.pack(fill=tk.BOTH, expand=True, padx=(0, 15), pady=(15, 0))
        tk.Frame(log_card, bg=ACCENT_PURPLE, height=3).pack(fill=tk.X)
        log_header = tk.Frame(log_card, bg=BG_CARD)
        log_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        tk.Label(log_header, text="Mini Log", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        self.log_text = scrolledtext.ScrolledText(log_card, height=8, font=FONT_OUTPUT, bg=SIDEBAR_BG, fg=WHITE, state=tk.DISABLED, borderwidth=0, highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Control buttons card (right column)
        controls_card = tk.Frame(bottom_panel, bg=BG_CARD)
        controls_card.pack(side=tk.LEFT, fill=tk.Y, padx=(15, 0))
        
        tk.Frame(controls_card, bg=ACCENT_ORANGE, height=3).pack(fill=tk.X)
        controls_header = tk.Frame(controls_card, bg=BG_CARD)
        controls_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        tk.Label(controls_header, text="Controls", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        
        controls_content = tk.Frame(controls_card, bg=BG_CARD)
        controls_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.start_btn = tk.Button(controls_content, text=f"{ICONS['start']} Start Classification", font=FONT_BTN, 
                                  fg=WHITE, bg=SUCCESS, activebackground=ACCENT_GREEN, activeforeground=WHITE, 
                                  bd=0, relief=tk.FLAT, cursor="hand2", command=self.toggle_prediction)
        self.start_btn.pack(fill=tk.X, pady=(0, 15))
        
        # Model selection UI inside Controls section
        tk.Label(controls_content, text="Model Selection", font=FONT_LABEL, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W, pady=(10, 8))
        # CNN
        tk.Label(controls_content, text="CNN Weights Folder", font=FONT_TEXT, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W)
        cnn_dir_row = tk.Frame(controls_content, bg=BG_CARD)
        cnn_dir_row.pack(fill=tk.X, pady=(2, 6))
        tk.Entry(cnn_dir_row, textvariable=self.cnn_models_dir, font=FONT_TEXT, fg=WHITE, bg=SIDEBAR_BG, insertbackground=WHITE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(cnn_dir_row, text=f"{ICONS['folder']}", font=FONT_BTN, fg=WHITE, bg=ACCENT_BLUE, bd=0, relief=tk.FLAT, cursor="hand2", command=self.browse_cnn_dir).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(controls_content, text="CNN Checkpoint (.pth)", font=FONT_TEXT, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W)
        self.cnn_dropdown = ttk.Combobox(controls_content, textvariable=self.cnn_model_var, state="readonly", width=28)
        self.cnn_dropdown.pack(fill=tk.X, pady=(2, 10))
        # YOLO
        tk.Label(controls_content, text="YOLO Weights Folder", font=FONT_TEXT, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W)
        yolo_dir_row = tk.Frame(controls_content, bg=BG_CARD)
        yolo_dir_row.pack(fill=tk.X, pady=(2, 6))
        tk.Entry(yolo_dir_row, textvariable=self.yolo_models_dir, font=FONT_TEXT, fg=WHITE, bg=SIDEBAR_BG, insertbackground=WHITE).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(yolo_dir_row, text=f"{ICONS['folder']}", font=FONT_BTN, fg=WHITE, bg=ACCENT_BLUE, bd=0, relief=tk.FLAT, cursor="hand2", command=self.browse_yolo_dir).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(controls_content, text="YOLO Weights (.pt)", font=FONT_TEXT, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W)
        self.yolo_dropdown = ttk.Combobox(controls_content, textvariable=self.yolo_model_var, state="readonly", width=28)
        self.yolo_dropdown.pack(fill=tk.X, pady=(2, 10))
        tk.Button(controls_content, text=f"{ICONS['refresh']} Refresh Models", font=FONT_BTN, fg=WHITE, bg=ACCENT_GREEN, bd=0, relief=tk.FLAT, cursor="hand2", command=self.refresh_model_lists).pack(fill=tk.X, pady=(8, 10))

        feedback_btn = tk.Button(controls_content, text=f"{ICONS['edit']} Give Feedback", font=FONT_BTN, 
                               fg=WHITE, bg=ACCENT_BLUE, activebackground=ACCENT_PURPLE, activeforeground=WHITE, 
                               bd=0, relief=tk.FLAT, cursor="hand2", command=self.feedback_dialog)
        feedback_btn.pack(fill=tk.X)

        # Populate model lists initially and schedule periodic refresh
        self.refresh_model_lists()
        self.schedule_model_refresh()

    def toggle_prediction(self):
        if not self.running:
            self.running = True
            self.start_btn.config(text=f"{ICONS['stop']} Stop Classification", bg=ERROR)
            # Show scrollbar when classification starts
            try:
                if not self.v_scrollbar.winfo_ismapped():
                    self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            except Exception:
                pass
            # Reset stop event and detection logs for a clean run
            try:
                self.stop_event.clear()
            except Exception:
                self.stop_event = threading.Event()
            self.detection_records = []
            self.detection_counts = {cat: 0 for cat in testing.categories}
            self.detection_conf_sums = {cat: 0.0 for cat in testing.categories}
            self._export_done = False
            try:
                self.after(0, self._update_counters_ui)
            except Exception:
                pass
            self.pred_thread = threading.Thread(target=self.run_prediction, daemon=True)
            self.pred_thread.start()
        else:
            # Signal prediction loop to stop and wait for cleanup
            try:
                self.stop_event.set()
            except Exception:
                pass
            self.running = False
            self.start_btn.config(text=f"{ICONS['start']} Start Classification", bg=SUCCESS)
            # Hide scrollbar when classification stops
            try:
                if self.v_scrollbar.winfo_ismapped():
                    self.v_scrollbar.pack_forget()
            except Exception:
                pass
            threading.Thread(target=self._finalize_after_stop, daemon=True).start()

    def run_prediction(self):
        # Resolve selected model paths
        cnn_dir = self.cnn_models_dir.get().strip()
        yolo_dir = self.yolo_models_dir.get().strip()
        cnn_name = self.cnn_model_var.get().strip()
        yolo_name = self.yolo_model_var.get().strip()
        cnn_path = os.path.join(cnn_dir, cnn_name) if cnn_dir and cnn_name else ''
        yolo_path = os.path.join(yolo_dir, yolo_name) if yolo_dir and yolo_name else ''

        self.ui_log(f"Loading model...\nCNN: {cnn_path if cnn_path else '(default)'}\nYOLO: {yolo_path if yolo_path else '(default)'}")

        def gui_callback(frame, labels, confidences, category_counts, roi):
            if not self.running:
                return
            # Apply fisheye and adjustments before display/prediction
            processed_frame = correct_fisheye(frame, enabled=self.fisheye_enabled.get())
            processed_frame = adjust_image(processed_frame, brightness=self.brightness.get(), contrast=self.contrast.get(), saturation=self.saturation.get())
            def update_ui():
                # Display frame
                frame_disp = cv2.resize(processed_frame, (480, 360))
                img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.webcam_label.imgtk = imgtk
                self.webcam_label.configure(image=imgtk)
                # Show all predictions
                if labels:
                    output_lines = []
                    for label, conf in zip(labels, confidences):
                        output_lines.append(f"Prediction: {label} | Confidence: {conf:.4f}")
                    self.prediction_text.set("\n".join(output_lines))
                    self.current_roi = roi
                    self.current_label = labels[0]
                    self.current_confidence = confidences[0]
                else:
                    self.prediction_text.set("No trash detected")
                    self.current_roi = None
                    self.current_label = None
                    self.current_confidence = None
                # Keep counter UI in sync with serial-based detection counts
                try:
                    self._update_counters_ui()
                except Exception:
                    pass
            self.after(0, update_ui)
        self.ui_log("Classification started...")
        try:
            testing.predict_real_time_with_box(
                num_boxes=1,
                callback=gui_callback,
                yolo_weights_path=yolo_path if os.path.isfile(yolo_path) else None,
                classifier_weights_path=cnn_path if os.path.isfile(cnn_path) else None,
                stop_event=self.stop_event,
                serial_callback=self.on_serial_send,
            )
        finally:
            self.ui_log("Completed.")
        self.running = False
        self.start_btn.config(text="Start Classification", bg=SUCCESS)
        # Hide scrollbar after completion
        try:
            if self.v_scrollbar.winfo_ismapped():
                self.v_scrollbar.pack_forget()
        except Exception:
            pass
        # Export reports automatically when the loop ends
        self._finalize_after_stop()

    def on_serial_send(self, trash_type: str, confidence: float):
        # Increment detection counters and store detailed record with timestamp
        try:
            ts = datetime.now()
            if trash_type in self.detection_counts:
                self.detection_counts[trash_type] += 1
                self.detection_conf_sums[trash_type] += float(confidence)
            else:
                # In case of 'unsupported' or unknown
                self.detection_counts[trash_type] = self.detection_counts.get(trash_type, 0) + 1
                self.detection_conf_sums[trash_type] = self.detection_conf_sums.get(trash_type, 0.0) + float(confidence)
            self.detection_records.append({
                'trash_type': trash_type,
                'confidence': float(confidence),
                'timestamp': ts,
            })
            # Update the UI counter labels to reflect serial-triggered detection
            try:
                self.after(0, self._update_counters_ui)
            except Exception:
                pass
        except Exception:
            pass

    def _finalize_after_stop(self):
        if self._export_done:
            return
        # Ensure prediction thread has exited
        try:
            if self.pred_thread is not None and self.pred_thread.is_alive():
                self.pred_thread.join(timeout=5.0)
        except Exception:
            pass
        # Export detection records to timestamped text files in database folder
        try:
            if not self.detection_records:
                return
            script_dir = testing.database_folder
            ts_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            detailed_path = os.path.join(script_dir, f"detailed_records_{ts_str}.txt")
            summary_path = os.path.join(script_dir, f"summary_records_{ts_str}.txt")
            # Detailed file
            with open(detailed_path, 'w', encoding='utf-8') as f:
                f.write("Trash          Count     Confidence Level    Time of Record\n")
                for rec in self.detection_records:
                    t = rec['trash_type']
                    c = rec['confidence']
                    tstamp = rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    line = f"{t:<13}  {1:<9}  {c:<18.2f}  {tstamp}\n"
                    f.write(line)
            # Summary file
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Trash          Total Count    Average Confidence\n")
                # Use categories order then any extras
                seen = set()
                for cat in testing.categories + [k for k in self.detection_counts.keys() if k not in testing.categories]:
                    if cat in seen:
                        continue
                    seen.add(cat)
                    total = int(self.detection_counts.get(cat, 0))
                    if total <= 0:
                        continue
                    sum_conf = float(self.detection_conf_sums.get(cat, 0.0))
                    avg_conf = (sum_conf / total) if total > 0 else 0.0
                    line = f"{cat:<13}  {total:<13}  {avg_conf:.2f}\n"
                    f.write(line)
            self.ui_log(f"Saved records to:\n  {detailed_path}\n  {summary_path}")
            self._export_done = True
        except Exception as e:
            self.ui_log(f"[Error] Exporting records failed: {e}")

    def ui_log(self, msg):
        try:
            if self.log_text is None:
                return
            def _append():
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
            self.after(0, _append)
        except Exception:
            pass

    def refresh_model_lists(self):
        # CNN models
        try:
            cnn_dir = self.cnn_models_dir.get().strip()
            cnn_files = []
            if cnn_dir and os.path.isdir(cnn_dir):
                cnn_files = [f for f in os.listdir(cnn_dir) if f.lower().endswith('.pth')]
            if self.cnn_dropdown is not None:
                self.cnn_dropdown['values'] = cnn_files
                if cnn_files:
                    if self.cnn_model_var.get() not in cnn_files:
                        self.cnn_model_var.set(cnn_files[0])
                else:
                    self.cnn_model_var.set('')
        except Exception as e:
            self.ui_log(f"[Warning] CNN model discovery failed: {e}")
        # YOLO models
        try:
            yolo_dir = self.yolo_models_dir.get().strip()
            yolo_files = []
            if yolo_dir and os.path.isdir(yolo_dir):
                yolo_files = [f for f in os.listdir(yolo_dir) if f.lower().endswith('.pt')]
            if self.yolo_dropdown is not None:
                self.yolo_dropdown['values'] = yolo_files
                if yolo_files:
                    if self.yolo_model_var.get() not in yolo_files:
                        self.yolo_model_var.set(yolo_files[0])
                else:
                    self.yolo_model_var.set('')
        except Exception as e:
            self.ui_log(f"[Warning] YOLO model discovery failed: {e}")

    def schedule_model_refresh(self):
        try:
            self.refresh_model_lists()
        except Exception:
            pass
        # Refresh every 5 seconds
        self.after(5000, self.schedule_model_refresh)

    def browse_cnn_dir(self):
        try:
            new_dir = filedialog.askdirectory(initialdir=self.cnn_models_dir.get() or os.getcwd(), title="Select CNN Weights Folder")
            if new_dir:
                self.cnn_models_dir.set(new_dir)
                self.refresh_model_lists()
                self.ui_log(f"CNN folder selected: {new_dir}")
        except Exception as e:
            self.ui_log(f"[Error] Selecting CNN folder failed: {e}")

    def browse_yolo_dir(self):
        try:
            new_dir = filedialog.askdirectory(initialdir=self.yolo_models_dir.get() or os.getcwd(), title="Select YOLO Weights Folder")
            if new_dir:
                self.yolo_models_dir.set(new_dir)
                self.refresh_model_lists()
                self.ui_log(f"YOLO folder selected: {new_dir}")
        except Exception as e:
            self.ui_log(f"[Error] Selecting YOLO folder failed: {e}")

    def feedback_dialog(self):
        if self.current_roi is None or not hasattr(self.current_roi, 'size') or self.current_roi.size == 0:
            messagebox.showinfo("Feedback", "No valid image to give feedback on yet.")
            return
        label = simpledialog.askstring("Feedback", f"Enter correct label for this image ({'/'.join(testing.categories)}):")
        if label is None:
            return
        label = label.strip().lower()
        if label not in testing.categories:
            messagebox.showerror("Feedback", f"Invalid label. Must be one of: {', '.join(testing.categories)}")
            return
        feedback_dir = os.path.join(testing.feedback_dataset_path, label)
        os.makedirs(feedback_dir, exist_ok=True)
        filename = f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(feedback_dir, filename)
        cv2.imwrite(filepath, self.current_roi)
        messagebox.showinfo("Feedback", f"Feedback saved: {filepath}")

    def reset_webcam_settings(self):
        self.fisheye_enabled.set(False)
        self.brightness.set(1.0)
        self.contrast.set(1.0)
        self.saturation.set(1.0)

    def stop_and_back(self):
        try:
            self.stop_event.set()
        except Exception:
            pass
        self.running = False
        self._finalize_after_stop()
        self.master.show_home()

# --- Database Page ---
class DatabasePage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_DARK)
        self.master = master
        self.selected_type = 'detailed'  # or 'summary'
        self.files = []
        self.create_layout()
        self.populate_file_list()

    def create_layout(self):
        # Sidebar for type selection and file list
        sidebar = tk.Frame(self, bg=SIDEBAR_BG, width=340)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Sidebar header
        tk.Label(sidebar, text="DATABASE", font=FONT_TITLE, fg=ACCENT_PURPLE, bg=SIDEBAR_BG).pack(pady=(30, 20))
        tk.Label(sidebar, text="Classification Records", font=FONT_TEXT, fg=GRAY_LIGHT, bg=SIDEBAR_BG).pack(pady=(0, 30))
        
        # Database type selection card
        type_card = tk.Frame(sidebar, bg=BG_CARD)
        type_card.pack(fill=tk.X, padx=18, pady=(0, 20))
        
        tk.Frame(type_card, bg=ACCENT_PURPLE, height=3).pack(fill=tk.X)
        type_content = tk.Frame(type_card, bg=BG_CARD)
        type_content.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(type_content, text="Record Type", font=FONT_LABEL, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 10))
        
        btn_type_frame = tk.Frame(type_content, bg=BG_CARD)
        btn_type_frame.pack(fill=tk.X, pady=(0, 10))
        self.btn_detailed = tk.Button(btn_type_frame, text=f"{ICONS['data']} Detailed Records", font=FONT_BTN, 
                                     fg=WHITE, bg=ACCENT_PURPLE if self.selected_type=='detailed' else GRAY_DARK, 
                                     activebackground=ACCENT_PURPLE, activeforeground=WHITE, bd=0, relief=tk.FLAT, 
                                     cursor="hand2", command=self.show_detailed)
        self.btn_detailed.pack(fill=tk.X, pady=(0, 5))
        self.btn_summary = tk.Button(btn_type_frame, text=f"{ICONS['chart']} Summary Records", font=FONT_BTN, 
                                   fg=WHITE, bg=ACCENT_PURPLE if self.selected_type=='summary' else GRAY_DARK, 
                                   activebackground=ACCENT_PURPLE, activeforeground=WHITE, bd=0, relief=tk.FLAT, 
                                   cursor="hand2", command=self.show_summary)
        self.btn_summary.pack(fill=tk.X)
        
        # File list
        tk.Label(sidebar, text="Available Files", font=FONT_LABEL, fg=GRAY_LIGHT, bg=SIDEBAR_BG).pack(anchor=tk.W, padx=18, pady=(0, 5))
        self.file_listbox = tk.Listbox(sidebar, width=38, font=FONT_OUTPUT, bg=BG_DARK, fg=WHITE, 
                                     selectbackground=ACCENT_PURPLE, borderwidth=0, highlightthickness=0)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 20))
        self.file_listbox.bind('<<ListboxSelect>>', self.display_file_content)
        
        # Back button
        SidebarButton(sidebar, "Back to Home", self.master.show_home, selected=True, icon=ICONS['home']).pack(fill=tk.X, pady=18, padx=18)

        # Main content area for file content
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # File content card
        content_card = tk.Frame(main, bg=BG_CARD)
        content_card.pack(fill=tk.BOTH, expand=True)
        
        tk.Frame(content_card, bg=ACCENT_BLUE, height=3).pack(fill=tk.X)
        content_header = tk.Frame(content_card, bg=BG_CARD)
        content_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        
        tk.Label(content_header, text="File Content Viewer", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        tk.Label(content_header, text="Select a file from the sidebar to view its contents", 
                font=FONT_TEXT, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W, pady=(5, 0))
        
        self.text_area = scrolledtext.ScrolledText(content_card, font=FONT_OUTPUT, bg=SIDEBAR_BG, fg=WHITE, 
                                                 state=tk.DISABLED, borderwidth=0, highlightthickness=0)
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def populate_file_list(self):
        # Look for files saved in the database folder
        script_dir = testing.database_folder
        if self.selected_type == 'detailed':
            pattern = os.path.join(script_dir, "detailed_records_*.txt")
        else:
            pattern = os.path.join(script_dir, "summary_records_*.txt")
        self.files = sorted(glob.glob(pattern), key=os.path.getctime, reverse=True)
        self.file_listbox.delete(0, tk.END)
        for f in self.files:
            self.file_listbox.insert(tk.END, os.path.basename(f))

    def show_detailed(self):
        self.selected_type = 'detailed'
        self.btn_detailed.config(bg=ACCENT_PURPLE)
        self.btn_summary.config(bg=GRAY_DARK)
        self.populate_file_list()
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)

    def show_summary(self):
        self.selected_type = 'summary'
        self.btn_detailed.config(bg=GRAY_DARK)
        self.btn_summary.config(bg=ACCENT_PURPLE)
        self.populate_file_list()
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)

    def display_file_content(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        file_path = self.files[idx]
        try:
            with open(file_path, "r") as f:
                content = f.read()
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, content)
            self.text_area.config(state=tk.DISABLED)
        except Exception as e:
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, f"Error reading file: {e}")
            self.text_area.config(state=tk.DISABLED)

# --- Retrain Menu ---
class RetrainMenu(tk.Frame):
    def __init__(self, master):
        print("RetrainMenu Loaded")
        super().__init__(master, bg=BG_DARK)
        self.master = master
        self.log_queue = queue.Queue()
        self.total_epochs = 20  # Default for retraining
        self.model_type = tk.StringVar(value="transfer")  # Add model type selection for retrain
        self.create_layout()
        self.training_log_updater()

    def create_layout(self):
        sidebar = tk.Frame(self, bg=SIDEBAR_BG, width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Sidebar header
        tk.Label(sidebar, text="RETRAINING", font=FONT_TITLE, fg=ACCENT_ORANGE, bg=SIDEBAR_BG).pack(pady=(30, 20))
        tk.Label(sidebar, text="Model Improvement", font=FONT_TEXT, fg=GRAY_LIGHT, bg=SIDEBAR_BG).pack(pady=(0, 30))
        SidebarButton(sidebar, "Back to Home", self.master.show_home, selected=True, icon=ICONS['home']).pack(fill=tk.X, pady=8, padx=18)
        
        # Status section
        status_section = tk.Frame(sidebar, bg=BG_CARD)
        status_section.pack(side=tk.BOTTOM, fill=tk.X, padx=18, pady=20)
        
        tk.Frame(status_section, bg=ACCENT_ORANGE, height=3).pack(fill=tk.X)
        status_content = tk.Frame(status_section, bg=BG_CARD)
        status_content.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(status_content, text="Status:", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W)
        self.status = tk.StringVar(value="Ready to retrain")
        tk.Label(status_content, textvariable=self.status, fg=WHITE, bg=BG_CARD, font=FONT_TEXT, wraplength=220).pack(anchor=tk.W, pady=(5, 0))

        main = tk.Frame(self, bg=BG_DARK)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Retraining configuration card
        config_card = tk.Frame(main, bg=BG_CARD)
        config_card.pack(fill=tk.X, pady=(0, 20))
        
        tk.Frame(config_card, bg=ACCENT_ORANGE, height=3).pack(fill=tk.X)
        config_content = tk.Frame(config_card, bg=BG_CARD)
        config_content.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(config_content, text="Retraining Configuration", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 15))
        
        # Model type selection (added)
        model_frame = tk.Frame(config_content, bg=BG_CARD)
        model_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(model_frame, text="Model Type", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 8))
        model_options = [
            ("Standard CNN", "cnn"),
            ("Transfer Learning (MobileNetV2)", "transfer"),
        ]
        for text, value in model_options:
            tk.Radiobutton(model_frame, text=text, variable=self.model_type, value=value, 
                          font=FONT_TEXT, fg=WHITE, bg=BG_CARD, selectcolor=ACCENT_ORANGE, 
                          activebackground=BG_CARD, activeforeground=ACCENT_ORANGE).pack(anchor=tk.W, padx=10)
        
        # Epochs info
        epochs_info = tk.Frame(config_content, bg=BG_CARD)
        epochs_info.pack(fill=tk.X, pady=(0, 15))
        tk.Label(epochs_info, text="Training Epochs", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 8))
        tk.Label(epochs_info, text=f"Fixed: {self.total_epochs} epochs", font=FONT_TEXT, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W, padx=10)
        
        # Model checkpoint selection for fine-tuning
        ckpt_frame = tk.Frame(config_content, bg=BG_CARD)
        ckpt_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(ckpt_frame, text="Checkpoint to Fine-Tune", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 8))
        self.ckpt_var = tk.StringVar(value="Auto: latest transfer")
        self.ckpt_dropdown = ttk.Combobox(ckpt_frame, textvariable=self.ckpt_var, width=48, state="readonly")
        try:
            ckpts = testing.discover_model_checkpoints()
            items = ["Auto: latest transfer"] + ckpts
        except Exception:
            items = ["Auto: latest transfer"]
        self.ckpt_dropdown['values'] = items
        if items:
            self.ckpt_dropdown.current(0)
        self.ckpt_dropdown.pack(anchor=tk.W, padx=10)

        # Start retraining button
        btn_frame = tk.Frame(config_content, bg=BG_CARD)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        tk.Button(btn_frame, text=f"{ICONS['retrain']} Start Retraining", font=FONT_BTN, fg=WHITE, bg=ACCENT_ORANGE, 
                 activebackground=ACCENT_GREEN, activeforeground=WHITE, bd=0, relief=tk.FLAT, 
                 width=18, height=2, cursor="hand2", command=run_in_thread(self.retrain_model)).pack(anchor=tk.W)
        
        # Training log card
        log_card = tk.Frame(main, bg=BG_CARD)
        log_card.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        tk.Frame(log_card, bg=ACCENT_GREEN, height=3).pack(fill=tk.X)
        log_header = tk.Frame(log_card, bg=BG_CARD)
        log_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        
        tk.Label(log_header, text="Retraining Progress", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(log_card, height=15, font=FONT_OUTPUT, bg=SIDEBAR_BG, fg=WHITE, 
                                                 state=tk.DISABLED, borderwidth=0, highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        # Progress tracking card
        progress_card = tk.Frame(main, bg=BG_CARD)
        progress_card.pack(fill=tk.X, pady=(0, 20))
        
        tk.Frame(progress_card, bg=ACCENT_ORANGE, height=3).pack(fill=tk.X)
        progress_content = tk.Frame(progress_card, bg=BG_CARD)
        progress_content.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(progress_content, text="Retraining Progress", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 15))
        
        # Preparation progress
        self.prep_progress_var = tk.DoubleVar()
        self.prep_label = tk.Label(progress_content, text="Data Preparation", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD)
        self.prep_label.pack(anchor=tk.W, pady=(0, 5))
        self.prep_progress_bar = ttk.Progressbar(progress_content, variable=self.prep_progress_var, maximum=1, length=500, mode='determinate')
        self.prep_progress_bar.pack(anchor=tk.W, pady=(0, 15))
        
        # Epoch progress
        self.epoch_label = tk.Label(progress_content, text="Epoch Progress", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD)
        self.epoch_label.pack(anchor=tk.W, pady=(0, 5))
        self.current_epoch_var = tk.StringVar(value="Current Epoch: -/-")
        self.current_epoch_label = tk.Label(progress_content, textvariable=self.current_epoch_var, font=FONT_TEXT, fg=ACCENT_ORANGE, bg=BG_CARD)
        self.current_epoch_label.pack(anchor=tk.W)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_content, variable=self.progress_var, maximum=self.total_epochs, length=500)
        self.progress_bar.pack(anchor=tk.W, pady=(5, 15))
        
        # Batch progress
        self.batch_label = tk.Label(progress_content, text="Batch Progress", font=FONT_LABEL, fg=GRAY_LIGHT, bg=BG_CARD)
        self.batch_label.pack(anchor=tk.W, pady=(0, 5))
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress_bar = ttk.Progressbar(progress_content, variable=self.batch_progress_var, maximum=1, length=500)
        self.batch_progress_bar.pack(anchor=tk.W)
        
        # Results visualization card
        results_card = tk.Frame(main, bg=BG_CARD)
        results_card.pack(fill=tk.BOTH, expand=True)
        
        tk.Frame(results_card, bg=ACCENT_PURPLE, height=3).pack(fill=tk.X)
        results_header = tk.Frame(results_card, bg=BG_CARD)
        results_header.pack(fill=tk.X, padx=20, pady=(15, 0))
        
        tk.Label(results_header, text="Retraining Results", font=FONT_SUBTITLE, fg=WHITE, bg=BG_CARD).pack(anchor=tk.W)
        
        # Panel for plots
        self.plot_panel = tk.Frame(results_card, bg=BG_CARD)
        self.plot_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.acc_img_label = tk.Label(self.plot_panel, bg=BG_CARD)
        self.acc_img_label.pack(side=tk.LEFT, padx=(0, 10))
        self.loss_img_label = tk.Label(self.plot_panel, bg=BG_CARD)
        self.loss_img_label.pack(side=tk.LEFT, padx=(10, 0))

    def set_status(self, msg):
        self.status.set(msg)

    def training_log_updater(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(200, self.training_log_updater)

    def retrain_model(self):
        self.set_status("Preparing to start retraining...")
        self.progress_var.set(0)
        self.batch_progress_var.set(0)
        self.prep_progress_var.set(0)
        self.current_epoch_var.set("Current Epoch: -/-")
        self.prep_progress_bar['maximum'] = len(testing.categories) if hasattr(testing, 'categories') else 1
        self.prep_progress_bar['value'] = 0
        self.prep_progress_bar.lift()  # Show preparation bar on top
        self.prep_label.lift()
        self.progress_bar.lower()      # Hide epoch bar initially
        self.epoch_label.lower()
        self.current_epoch_label.lower()
        self.batch_progress_bar.lower()# Hide batch bar initially
        self.batch_label.lower()
        def run():
            def progress_callback(kind, current, total=None):
                if kind == 'preparing':
                    self.prep_progress_bar['maximum'] = total
                    self.prep_progress_var.set(current)
                    self.set_status(f"Preparing data: {current}/{total}...")
                    # Show prep bar, hide others
                    self.prep_progress_bar.lift()
                    self.prep_label.lift()
                    self.progress_bar.lower()
                    self.epoch_label.lower()
                    self.current_epoch_label.lower()
                    self.batch_progress_bar.lower()
                    self.batch_label.lower()
                elif kind == 'epoch':
                    self.progress_bar['maximum'] = total
                    self.progress_var.set(current)
                    self.current_epoch_var.set(f"Current Epoch: {current}/{total}")
                    # Show epoch bar, hide prep bar, show batch bar
                    self.prep_progress_bar.lower()
                    self.prep_label.lower()
                    self.progress_bar.lift()
                    self.epoch_label.lift()
                    self.current_epoch_label.lift()
                    self.batch_progress_bar.lift()
                    self.batch_label.lift()
                    self.set_status(f"Retraining: Epoch {current}/{total}")
                elif kind == 'batch':
                    self.batch_progress_bar['maximum'] = total
                    self.batch_progress_var.set(current)
                elif kind == 'status':
                    if current == 'preparing':
                        self.set_status("Preparing for retraining...")
                    elif current == 'training':
                        self.set_status("Retraining in progress...")
                    elif current == 'done':
                        self.set_status("Retraining complete.")
            try:
                # Determine checkpoint choice
                selected = self.ckpt_var.get().strip() if hasattr(self, 'ckpt_var') else "Auto: latest transfer"
                if selected and not selected.lower().startswith("auto"):
                    inferred = testing.infer_model_type_from_path(selected)
                    model_type = inferred
                    init_path = selected
                else:
                    model_type = self.model_type.get()  # Use selected radio button value
                    init_path = None  # falls back to trash_classifier_transfer_improved.pth

                # Call PyTorch training script externally
                testing.retrain_model(
                    epochs=self.total_epochs,
                    log_queue=self.log_queue,
                    progress_callback=progress_callback,
                    model_type=model_type,
                    initial_weights_path=init_path
                )
                self.set_status("Retraining complete.")
                self.show_training_plots()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.set_status(f"Error: {e}")
        threading.Thread(target=run, daemon=True).start()

    def show_training_plots(self):
        import os
        from PIL import Image, ImageTk
        acc_path = os.path.join(testing.database_folder, 'training_plot.png')
        loss_path = os.path.join(testing.database_folder, 'loss_plot.png')
        if os.path.exists(acc_path):
            acc_img = Image.open(acc_path)
            acc_img = acc_img.resize((320, 240))
            acc_imgtk = ImageTk.PhotoImage(acc_img)
            self.acc_img_label.imgtk = acc_imgtk
            self.acc_img_label.config(image=acc_imgtk, text="")
        else:
            self.acc_img_label.config(image="", text="")
        if os.path.exists(loss_path):
            loss_img = Image.open(loss_path)
            loss_img = loss_img.resize((320, 240))
            loss_imgtk = ImageTk.PhotoImage(loss_img)
            self.loss_img_label.imgtk = loss_imgtk
            self.loss_img_label.config(image=loss_imgtk, text="")
        else:
            self.loss_img_label.config(image="", text="")

# --- Modern Card Button ---
class ModernButton(tk.Frame):
    def __init__(self, parent, text, command, width=280, height=60, 
                 bg_color=BG_CARD, accent_color=ACCENT_GREEN, text_color=WHITE, icon=None):
        super().__init__(parent, bg=BG_DARK, width=width, height=height)
        self.pack_propagate(False)
        self.command = command
        self.bg_color = bg_color
        self.accent_color = accent_color
        self.text_color = text_color
        self.icon = icon
        
        # Create the main button frame with rounded appearance
        self.btn_frame = tk.Frame(self, bg=bg_color, height=height-4)
        self.btn_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.btn_frame.pack_propagate(False)
        
        # Add accent border on the left
        self.accent_border = tk.Frame(self.btn_frame, bg=accent_color, width=4)
        self.accent_border.pack(side=tk.LEFT, fill=tk.Y)
        
        # Main content area
        self.content_frame = tk.Frame(self.btn_frame, bg=bg_color)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=12)
        
        # Icon and text layout
        if icon:
            # Icon label
            self.icon_label = tk.Label(
                self.content_frame, text=icon, font=FONT_ICON_LARGE,
                fg=accent_color, bg=bg_color, cursor="hand2"
            )
            self.icon_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Text label
            self.btn_label = tk.Label(
                self.content_frame, text=text, font=FONT_BTN,
                fg=text_color, bg=bg_color, cursor="hand2"
            )
            self.btn_label.pack(side=tk.LEFT, anchor=tk.W)
        else:
            # Button text only
            self.btn_label = tk.Label(
                self.content_frame, text=text, font=FONT_BTN,
                fg=text_color, bg=bg_color, cursor="hand2"
            )
            self.btn_label.pack(anchor=tk.W)
        
        # Bind click events to all components
        self._bind_events()
        
    def _bind_events(self):
        components = [self, self.btn_frame, self.content_frame, self.btn_label]
        if hasattr(self, 'icon_label'):
            components.append(self.icon_label)
        for comp in components:
            comp.bind("<Button-1>", self._on_click)
            comp.bind("<Enter>", self._on_enter)
            comp.bind("<Leave>", self._on_leave)
    
    def _on_enter(self, event):
        self.btn_frame.config(bg=HOVER_BG)
        self.content_frame.config(bg=HOVER_BG)
        self.btn_label.config(bg=HOVER_BG, fg=self.accent_color)
        if hasattr(self, 'icon_label'):
            self.icon_label.config(bg=HOVER_BG)
    
    def _on_leave(self, event):
        self.btn_frame.config(bg=self.bg_color)
        self.content_frame.config(bg=self.bg_color)
        self.btn_label.config(bg=self.bg_color, fg=self.text_color)
        if hasattr(self, 'icon_label'):
            self.icon_label.config(bg=self.bg_color)
    
    def _on_click(self, event):
        if callable(self.command):
            self.command()

# --- Legacy LimbusButton (kept for compatibility) ---
class LimbusButton(tk.Canvas):
    def __init__(self, parent, text, command, width=200, height=40, 
                 bg=BG_CARD, fg=ACCENT_GREEN, active_bg=ACCENT_GREEN, active_fg=BG_DARK):
        super().__init__(parent, width=width, height=height, bg=BG_DARK, 
                        highlightthickness=0, bd=0)
        self.command = command
        self._draw_button(text, bg, fg)
        self.bind("<Enter>", lambda e: self._draw_button(text, active_bg, active_fg))
        self.bind("<Leave>", lambda e: self._draw_button(text, bg, fg))
        self.bind("<Button-1>", self._on_click)
        self.tag_bind("all", "<Button-1>", self._on_click)
    def _draw_button(self, text, bg, fg):
        self.delete("all")
        self.create_rectangle(2, 2, self.winfo_reqwidth()-2, self.winfo_reqheight()-2, 
                            outline="", fill=bg, width=0, tags="all")
        self.create_text(self.winfo_reqwidth()//2, self.winfo_reqheight()//2, 
                        text=text, font=FONT_BTN, fill=fg, tags="all")
    def _on_click(self, event=None):
        if callable(self.command):
            self.command()

if __name__ == "__main__":
    app = ThrowawayApp()
    app.mainloop() 