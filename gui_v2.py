import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from model_architecture import NWMv1
from utilities import (
    load_all_data, organise_dataset, get_train_list,
    load_vocab_embedding, load_pretrained_model
)

#loads all data and model required for gui
def load_data():
    dataset = organise_dataset(load_all_data())
    num_sentences_in_vocab_list = 500
    num_samples = 895
    embedding_dim = 150
    device = 'cpu'
    
    #vocab list for compression
    _, _, _, vocab_list, vocab_dict = get_train_list(
        train_size=num_sentences_in_vocab_list, dataset=dataset
    )
    
    vocab_path = r"demo_weights_metrics\vocab_and_model\vocab_embedding.pt"
    vocab_embedding, _ = load_vocab_embedding(vocab_path, device, normalise=False)
    
    model = NWMv1(
        vocab_list=vocab_list,
        vocab_embedding=vocab_embedding,
        embedding_size=embedding_dim
    )
    
    model, _ = load_pretrained_model(model, r"current_best.pt", device)
    
    #full dataset samples
    train_tensor, sentence_id_tensor, sentence_mapping, _, _ = get_train_list(
        train_size=num_samples, dataset=dataset
    )
    
    #load sentence transformer for semantic similarity
    sentence_model = SentenceTransformer('all-mpnet-base-v2')
    
    return model, train_tensor, sentence_id_tensor, sentence_mapping, device, sentence_model

#main gui application
class EEGVisualisationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Real-Time Decoding Visualisation")
        self.root.configure(bg='#1a1a1c')
        
        #load data and model
        self.model, self.train_tensor, self.sentence_id_tensor, self.sentence_mapping, self.device, self.sentence_model = load_data()
        
        #current sample state
        self.current_sample_idx = 0
        self.current_eeg = None #(channels, timepoints)
        self.cursor_position = 0
        self.true_embedding = None #precomputed embedding for true sentence
        
        #playback state
        self.segment_length = 500
        self.is_playing = False
        self.timer_id = None
        self.playback_position = 0
        self.buffer = None #(channels, segment_length)
        self.buffer_fill_count = 0
        
        #model states across windows
        self.c_state = None
        self.p_state = None
        self.prev_output = None
        
        #sentence display widgets
        self.true_text = None
        self.pred_text = None
        self.true_wordcount_label = None
        self.pred_wordcount_label = None
        
        #evaluation display widgets
        self.windows_label = None
        self.overlap_label = None
        self.similarity_label = None
        self.conf_fig = None
        self.conf_ax = None
        self.conf_canvas = None
        self.conf_colorbar = None
        
        #network visualisation components
        self.viz_fig = None
        self.viz_axes = None #array of 6 axes for heatmaps
        self.viz_canvas = None
        
        #individual min max ranges for each network state visualisation
        self.viz_limits = [
            (-3, 7),   #state_t
            (-3, 7), #pre_activations
            (-1.5, 1),   #post_activations
            (-3, 6),   #cognitive_signals
            (-3, 6),   #motor_prop
            (-1, 1)    #policy_state
        ]
        
        #custom colormaps
        self.conf_cmap = 'RdYlGn' #red-yellow-green where green is high
        # self.network_cmap = LinearSegmentedColormap.from_list('network', ['#6A0DFF', '#D916F8', '#FF9980'])
        self.network_cmap = LinearSegmentedColormap.from_list('network', ['#6A0DFF', '#FF9980'])
        
        #build gui
        self.create_widgets()
        
        #load first sample
        self.load_sample(0)
    
    #creates all gui widgets and layout
    def create_widgets(self):
        #main container split into left and right panels
        left_panel = tk.Frame(self.root, bg='#1a1a1c')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = tk.Frame(self.root, width=400, bg='#1a1a1c')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_panel.pack_propagate(False)
        
        #sample selection dropdown
        top_frame = tk.Frame(left_panel, bg='#1a1a1c')
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        tk.Label(top_frame, text="Select Sample:", bg='#1a1a1c', fg='white').pack(side=tk.LEFT)
        
        self.sample_var = tk.StringVar()
        sample_indices = [str(i) for i in range(len(self.train_tensor))]
        self.sample_dropdown = ttk.Combobox(
            top_frame, 
            textvariable=self.sample_var, 
            values=sample_indices,
            state='readonly',
            width=10
        )
        self.sample_dropdown.current(0)
        self.sample_dropdown.pack(side=tk.LEFT, padx=5)
        self.sample_dropdown.bind('<<ComboboxSelected>>', self.on_sample_select)
        
        #eeg plot area
        plot_frame = tk.Frame(left_panel, bg='#1a1a1c')
        plot_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.fig = Figure(figsize=(10, 6), facecolor='#1a1a1c')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1a1c')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.cursor_line = None #vertical line indicating playback position
        
        #cursor position slider
        slider_frame = tk.Frame(left_panel, bg='#1a1a1c')
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        tk.Label(slider_frame, text="Start Position:", bg='#1a1a1c', fg='white').pack(side=tk.LEFT)
        
        self.cursor_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.update_cursor_position,
            bg='#3a3a3c',
            fg='white',
            troughcolor='#2a2a2c',
            highlightbackground='#1a1a1c'
        )
        self.cursor_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        #play and pause controls
        control_frame = tk.Frame(left_panel, bg='#1a1a1c')
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.play_button = tk.Button(
            control_frame, 
            text="Play", 
            command=self.on_play,
            bg='#3a3a3c',
            fg='white',
            activebackground='#4a4a4c',
            activeforeground='white'
        )
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = tk.Button(
            control_frame, 
            text="Pause", 
            command=self.on_pause,
            bg='#3a3a3c',
            fg='white',
            activebackground='#4a4a4c',
            activeforeground='white'
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        #true and predicted sentence display
        sentence_frame = tk.LabelFrame(
            left_panel, 
            text="Sentence Decoding", 
            font=('Arial', 10, 'bold'),
            bg='#1a1a1c',
            fg='white'
        )
        sentence_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=10)
        
        #true sentence row
        true_row = tk.Frame(sentence_frame, bg='#1a1a1c')
        true_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Label(
            true_row, 
            text="True:", 
            font=('Arial', 10, 'bold'), 
            width=6, 
            anchor='w',
            bg='#1a1a1c',
            fg='white'
        ).pack(side=tk.LEFT)
        
        self.true_text = tk.Text(
            true_row,
            height=2,
            width=60,
            font=('Arial', 10),
            bg='#2a2a2c',
            fg='white',
            relief=tk.SUNKEN,
            borderwidth=2,
            wrap=tk.WORD,
            state=tk.DISABLED,
            insertbackground='white'
        )
        self.true_text.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.true_wordcount_label = tk.Label(
            true_row,
            text="Words: 0/16",
            font=('Arial', 9),
            width=12,
            anchor='e',
            bg='#1a1a1c',
            fg='white'
        )
        self.true_wordcount_label.pack(side=tk.RIGHT)
        
        #predicted sentence row
        pred_row = tk.Frame(sentence_frame, bg='#1a1a1c')
        pred_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Label(
            pred_row, 
            text="Pred:", 
            font=('Arial', 10, 'bold'), 
            width=6, 
            anchor='w',
            bg='#1a1a1c',
            fg='white'
        ).pack(side=tk.LEFT)
        
        self.pred_text = tk.Text(
            pred_row,
            height=2,
            width=60,
            font=('Arial', 10),
            bg='#2a2a2c',
            fg='white',
            relief=tk.SUNKEN,
            borderwidth=2,
            wrap=tk.WORD,
            state=tk.DISABLED,
            insertbackground='white'
        )
        self.pred_text.tag_config('prediction', foreground='cyan')
        self.pred_text.tag_config('final', foreground='lightgreen', font=('Arial', 10, 'bold'))
        self.pred_text.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.pred_wordcount_label = tk.Label(
            pred_row,
            text="Words: 0/16",
            font=('Arial', 9),
            width=12,
            anchor='e',
            bg='#1a1a1c',
            fg='white'
        )
        self.pred_wordcount_label.pack(side=tk.RIGHT)
        
        #evaluation metrics row
        eval_row = tk.Frame(sentence_frame, bg='#1a1a1c')
        eval_row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.windows_label = tk.Label(
            eval_row, 
            text="Windows: 0", 
            font=('Arial', 9), 
            anchor='w',
            bg='#1a1a1c',
            fg='white'
        )
        self.windows_label.pack(side=tk.LEFT, padx=10)
        
        self.similarity_label = tk.Label(
            eval_row, 
            text="Similarity: 0.00", 
            font=('Arial', 9), 
            anchor='w',
            bg='#1a1a1c',
            fg='white'
        )
        self.similarity_label.pack(side=tk.LEFT, padx=10)
        
        self.overlap_label = tk.Label(
            eval_row, 
            text="Overlap: {}", 
            font=('Arial', 9), 
            anchor='w',
            bg='#1a1a1c',
            fg='white'
        )
        self.overlap_label.pack(side=tk.LEFT, padx=10)
        
        #confidence heatmap
        conf_frame = tk.Frame(sentence_frame, bg='#1a1a1c')
        conf_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Label(
            conf_frame, 
            text="Confidence:", 
            font=('Arial', 9, 'bold'),
            bg='#1a1a1c',
            fg='white'
        ).pack(side=tk.TOP, anchor='w')
        
        self.conf_fig = Figure(figsize=(8, 1.0), facecolor='#1a1a1c')
        self.conf_ax = self.conf_fig.add_subplot(111)
        self.conf_ax.set_facecolor('#1a1a1c')
        self.conf_ax.axis('off')
        self.conf_fig.tight_layout(pad=0)
        
        self.conf_canvas = FigureCanvasTkAgg(self.conf_fig, master=conf_frame)
        self.conf_canvas.get_tk_widget().pack(fill=tk.X)
        
        #network states visualisation on right panel
        states_frame = tk.LabelFrame(
            right_panel, 
            text="Network States Visualisation", 
            font=('Arial', 10, 'bold'),
            bg='#1a1a1c',
            fg='white'
        )
        states_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        #create matplotlib figure with 6 subplots for heatmaps
        self.viz_fig = Figure(figsize=(5, 10), facecolor='#1a1a1c')
        
        #height ratios: vectors=1, matrices=3
        height_ratios = [1, 3, 1, 3, 3, 1]
        gs = GridSpec(6, 1, figure=self.viz_fig, height_ratios=height_ratios)
        
        #create 6 subplots with custom heights
        self.viz_axes = []
        titles = [
            'State T (Feature Extraction)',
            'Pre-Activations (Cognitive)',
            'Post-Activations (Cognitive)',
            'Cognitive Signals',
            'Motor Propagation',
            'Policy State'
        ]
        
        for i in range(6):
            ax = self.viz_fig.add_subplot(gs[i])
            ax.set_title(titles[i], fontsize=9, pad=5, color='white')
            ax.set_facecolor('#1a1a1c')
            ax.tick_params(colors='white', labelsize=7)
            self.viz_axes.append(ax)
        
        self.viz_fig.tight_layout(pad=2.0)
        
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=states_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    #handles sample selection from dropdown
    def on_sample_select(self, event):
        selected_idx = int(self.sample_var.get())
        self.load_sample(selected_idx)
    
    #loads selected eeg sample and displays it
    def load_sample(self, sample_idx):
        self.current_sample_idx = sample_idx
        self.current_eeg = self.train_tensor[sample_idx, :, :].numpy() #(channels, timepoints)
        
        #get true sentence from mapping
        sentence_id = self.sentence_id_tensor[sample_idx].item()
        true_words = self.sentence_mapping[sentence_id]
        true_sentence = ' '.join(true_words)
        
        #update true sentence display
        self.true_text.config(state=tk.NORMAL)
        self.true_text.delete('1.0', tk.END)
        self.true_text.insert('1.0', true_sentence)
        self.true_text.config(state=tk.DISABLED)
        
        #update true word count
        true_word_count = len(true_words)
        self.true_wordcount_label.config(text=f"Words: {true_word_count}/16")
        
        #precompute true sentence embedding
        self.true_embedding = self.sentence_model.encode(true_sentence, convert_to_tensor=True)
        
        #reset cursor to start
        self.cursor_position = 0
        self.cursor_slider.config(to=self.current_eeg.shape[1] - 1)
        self.cursor_slider.set(0)
        
        #clear prediction display
        self.pred_text.config(state=tk.NORMAL)
        self.pred_text.delete('1.0', tk.END)
        self.pred_text.config(state=tk.DISABLED)
        self.pred_wordcount_label.config(text="Words: 0/16")
        
        #clear evaluation displays
        self.windows_label.config(text="Windows: 0")
        self.similarity_label.config(text="Similarity: 0.00")
        self.overlap_label.config(text="Overlap: {}")
        
        #clear confidence heatmap
        self.conf_ax.clear()
        self.conf_ax.axis('off')
        self.conf_canvas.draw()
        
        #reset playback state
        self.reset_playback_state()
        
        #render eeg plot
        self.plot_eeg()
    
    #plots eeg with stacked channels and cursor line
    def plot_eeg(self):
        self.ax.clear()
        
        num_channels, num_timepoints = self.current_eeg.shape
        
        #normalise each channel to zero mean and unit variance for consistent visualisation
        normalised_eeg = np.zeros_like(self.current_eeg)
        for ch_idx in range(num_channels):
            channel_data = self.current_eeg[ch_idx, :]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                normalised_eeg[ch_idx, :] = (channel_data - mean) / std
            else:
                normalised_eeg[ch_idx, :] = channel_data - mean
        
        #stack channels with fixed spacing
        spacing = 3
        for ch_idx in range(num_channels):
            offset = ch_idx * spacing
            self.ax.plot(normalised_eeg[ch_idx, :] + offset, linewidth=0.3, color='darkred', alpha=0.7)
        
        #draw cursor line at current position
        self.cursor_line = self.ax.axvline(x=self.cursor_position, color='blue', linewidth=2, label='Cursor')
        
        self.ax.set_xlabel('Timepoints')
        self.ax.set_ylabel('Channel Index (scaled)')
        self.ax.set_yticks([i * spacing for i in range(0, num_channels, 20)])
        self.ax.set_yticklabels([str(i) for i in range(0, num_channels, 20)])
        self.ax.set_title(f'Sample {self.current_sample_idx} - EEG Recording ({num_channels} channels)')
        self.ax.legend()
        
        #set axis colors to white
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        
        self.canvas.draw()
    
    #updates cursor position from slider movement
    def update_cursor_position(self, value):
        self.cursor_position = int(float(value))
        
        #move cursor line on plot
        if self.cursor_line is not None:
            self.cursor_line.set_xdata([self.cursor_position, self.cursor_position])
            self.canvas.draw()
    
    #resets all playback and model state
    def reset_playback_state(self):
        self.is_playing = False
        if self.timer_id is not None:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
        self.playback_position = 0
        self.buffer_fill_count = 0
        self.buffer = None
        self.c_state = None
        self.p_state = None
        self.prev_output = None
    
    #initialises buffer and model states for playback
    def initialise_playback(self):
        num_channels = self.current_eeg.shape[0]
        self.buffer = np.zeros((num_channels, self.segment_length), dtype=np.float32)
        self.buffer_fill_count = 0
        self.playback_position = self.cursor_position
        self.c_state = None
        self.p_state = None
        self.prev_output = None
    
    #starts playback from cursor position
    def on_play(self):
        if self.is_playing:
            return
        
        self.is_playing = True
        self.initialise_playback()
        self.play_button.config(state='disabled')
        self.pause_button.config(state='normal')
        self.sample_dropdown.config(state='disabled')
        self.cursor_slider.config(state='disabled')
        
        #start timer callback
        self.playback_step()
    
    #stops playback and resets buffer
    def on_pause(self):
        if not self.is_playing:
            return
        
        self.is_playing = False
        if self.timer_id is not None:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
        
        #clear buffer on pause
        self.buffer_fill_count = 0
        
        self.play_button.config(state='normal')
        self.pause_button.config(state='disabled')
        self.sample_dropdown.config(state='normal')
        self.cursor_slider.config(state='normal')
    
    #single step of playback timer callback
    def playback_step(self):
        if not self.is_playing:
            return
        
        num_timepoints = self.current_eeg.shape[1]
        batch_size = 100 #increase/decrease this to control playback speed vs smoothness
        
        #process multiple timepoints per gui update
        for _ in range(batch_size):
            #check if we've reached end of recording
            if self.playback_position >= num_timepoints:
                #trigger final reasoning before stopping
                self.trigger_final_reasoning()
                self.on_pause()
                return
            
            #accumulate current timepoint into buffer
            self.buffer[:, self.buffer_fill_count] = self.current_eeg[:, self.playback_position]
            self.buffer_fill_count += 1
            self.playback_position += 1
            
            #check if buffer is full or we've reached end
            should_process = (self.buffer_fill_count == self.segment_length) or (self.playback_position >= num_timepoints)
            
            if should_process:
                self.process_window()
        
        #update cursor visual once per batch
        self.cursor_position = self.playback_position
        if self.cursor_line is not None:
            self.cursor_line.set_xdata([self.cursor_position, self.cursor_position])
            self.canvas.draw_idle()
        
        #schedule next batch
        if self.is_playing:
            self.timer_id = self.root.after(2, self.playback_step) #2ms per timepoint
    
    #processes current buffer window through model
    def process_window(self):
        #zero-pad if buffer not full
        if self.buffer_fill_count < self.segment_length:
            self.buffer[:, self.buffer_fill_count:] = 0
        
        #prepare input tensor (1, channels, segment_length)
        window = torch.from_numpy(self.buffer).unsqueeze(0).to(self.device)
        
        #forward pass through model with state collection
        state_t = self.model.extract_features(window)
        c_signals, self.c_state = self.model.think(state_t, self.c_state, prev_output=self.prev_output)
        mu, log_sig, self.p_state, motor_prop = self.model.propagate_action(c_signals, self.p_state, return_prop=True)
        self.prev_output = mu
        
        #decode prediction with confidences
        det_sen, det_confs, det_conf = self.model.decode_vocab_ids(mu)
        predicted_sentence = self.model.construct_sentence(det_sen)
        
        #update prediction display with colored text
        self.pred_text.config(state=tk.NORMAL)
        self.pred_text.delete('1.0', tk.END)
        self.pred_text.insert('1.0', predicted_sentence, 'prediction')
        self.pred_text.config(state=tk.DISABLED)
        
        #update predicted word count
        pred_word_count = len(predicted_sentence.split())
        self.pred_wordcount_label.config(text=f"Words: {pred_word_count}/16")
        
        #update windows counted
        windows_counted = self.model.windows_counted
        self.windows_label.config(text=f"Windows: {windows_counted}")
        
        #compute semantic similarity
        pred_embedding = self.sentence_model.encode(predicted_sentence, convert_to_tensor=True)
        similarity = F.cosine_similarity(self.true_embedding.unsqueeze(0), pred_embedding.unsqueeze(0)).item()
        self.similarity_label.config(text=f"Similarity: {similarity:.4f}")
        
        #compute word overlap
        true_sentence = self.true_text.get('1.0', tk.END).strip()
        true_words_set = set(true_sentence.split())
        pred_words_set = set(predicted_sentence.split())
        overlap = true_words_set.intersection(pred_words_set)
        self.overlap_label.config(text=f"Overlap: {overlap}")
        
        #update confidence heatmap
        self.update_confidence_heatmap(det_confs)
        
        #update network state visualisations
        self.update_visualisations(state_t, self.c_state, c_signals, motor_prop, self.p_state)
        
        #clear buffer for next window
        self.buffer_fill_count = 0
    
    #performs final reasoning after all windows processed
    def trigger_final_reasoning(self):
        #get final state from model
        final_state = self.model.get_final_state()
        
        #final forward pass with state collection
        c_signals, self.c_state = self.model.think(final_state, self.c_state, prev_output=self.prev_output)
        mu, log_sig, self.p_state, motor_prop = self.model.propagate_action(c_signals, self.p_state, return_prop=True)
        
        #decode final prediction with confidences
        det_sen, det_confs, det_conf = self.model.decode_vocab_ids(mu)
        final_sentence = self.model.construct_sentence(det_sen)
        
        #update prediction display with final marker
        self.pred_text.config(state=tk.NORMAL)
        self.pred_text.delete('1.0', tk.END)
        self.pred_text.insert('1.0', f"FINAL: {final_sentence}", 'final')
        self.pred_text.config(state=tk.DISABLED)
        
        #update predicted word count
        pred_word_count = len(final_sentence.split())
        self.pred_wordcount_label.config(text=f"Words: {pred_word_count}/16")
        
        #update windows counted
        windows_counted = self.model.windows_counted
        self.windows_label.config(text=f"Windows: {windows_counted}")
        
        #compute semantic similarity
        pred_embedding = self.sentence_model.encode(final_sentence, convert_to_tensor=True)
        similarity = F.cosine_similarity(self.true_embedding.unsqueeze(0), pred_embedding.unsqueeze(0)).item()
        self.similarity_label.config(text=f"Similarity: {similarity:.4f}")
        
        #compute word overlap
        true_sentence = self.true_text.get('1.0', tk.END).strip()
        true_words_set = set(true_sentence.split())
        pred_words_set = set(final_sentence.split())
        overlap = true_words_set.intersection(pred_words_set)
        self.overlap_label.config(text=f"Overlap: {overlap}")
        
        #update confidence heatmap
        self.update_confidence_heatmap(det_confs)
        
        #update network state visualisations
        self.update_visualisations(final_state, self.c_state, c_signals, motor_prop, self.p_state)
    
    #updates confidence heatmap visualisation
    def update_confidence_heatmap(self, det_confs):
        #extract confidence vector
        conf_np = det_confs.squeeze(0).detach().cpu().numpy() #(thought_steps,)
        conf_display = conf_np.reshape(1, -1) #(1, thought_steps)
        
        #clear and redraw
        self.conf_ax.clear()
        self.conf_ax.set_facecolor('#1a1a1c')
        
        #plot heatmap
        im = self.conf_ax.imshow(conf_display, aspect='auto', cmap=self.conf_cmap, vmin=-2, vmax=1)
        self.conf_ax.axis('off')
        
        #add colorbar if not exists
        if self.conf_colorbar is None:
            self.conf_colorbar = self.conf_fig.colorbar(im, ax=self.conf_ax, orientation='horizontal', pad=0.1)
            self.conf_colorbar.set_ticks([-2, -0.5, 1])
            self.conf_colorbar.set_ticklabels(['Low', 'Mid', 'High'])
            self.conf_colorbar.ax.tick_params(labelsize=7, colors='white')
        
        self.conf_canvas.draw_idle()

    #uses dynamic min max from each state for heatmaps
    def update_visualisations(self, state_t, c_state, c_signals, motor_prop, p_state):
        #extract state_t as 1d array
        state_t_np = state_t.squeeze(0).detach().cpu().numpy() #(state_dim,)
        state_t_display = state_t_np.reshape(1, -1) #(1, state_dim)
        
        #extract cognitive pre-activations history and transpose for consistent axes
        pre_activations = c_state['pre_activation_history'].squeeze(0).detach().cpu().numpy() #(neurons, memory_length)
        pre_activations = pre_activations.T #(memory_length, neurons)
        
        #extract cognitive post-activations
        post_activations = c_state['post_activations'].squeeze(0).detach().cpu().numpy() #(neurons,)
        post_activations_display = post_activations.reshape(1, -1) #(1, neurons)
        
        #extract cognitive signals and transpose for consistent axes
        c_signals_np = c_signals.squeeze(0).detach().cpu().numpy() #(signals, thought_steps)
        c_signals_np = c_signals_np.T #(thought_steps, signals)
        
        #extract motor propagation
        motor_prop_np = motor_prop.squeeze(0).detach().cpu().numpy() #(thought_steps, neurons)
        
        #extract policy state
        p_state_np = p_state.squeeze(0).detach().cpu().numpy() #(neurons,)
        p_state_display = p_state_np.reshape(1, -1) #(1, neurons)
        
        #data array for plotting
        data_list = [
            state_t_display,
            pre_activations,
            post_activations_display,
            c_signals_np,
            motor_prop_np,
            p_state_display
        ]
        
        #titles for each subplot
        titles = [
            'State T (Feature Extraction)',
            'Pre-Activations (Cognitive)',
            'Post-Activations (Cognitive)',
            'Cognitive Signals',
            'Motor Propagation',
            'Policy State'
        ]
        
        #shape labels for each visualisation with consistent axes
        shape_labels = [
            f'x: state_dim={state_t_display.shape[1]}',
            f'x: neurons={pre_activations.shape[1]}, y: memory={pre_activations.shape[0]}',
            f'x: neurons={post_activations_display.shape[1]}',
            f'x: signals={c_signals_np.shape[1]}, y: thought_steps={c_signals_np.shape[0]}',
            f'x: neurons={motor_prop_np.shape[1]}, y: thought_steps={motor_prop_np.shape[0]}',
            f'x: neurons={p_state_display.shape[1]}'
        ]
        
        #update each subplot
        for i, (ax, data, shape_label, title) in enumerate(zip(self.viz_axes, data_list, shape_labels, titles)):
            ax.clear()
            ax.set_facecolor('#1a1a1c')
            
            #vmin, vmax = self.viz_limits[i]  #uncomment to use fixed limits
            vmin, vmax = data.min(), data.max()  #dynamic limits from data
            ax.imshow(data, aspect='auto', cmap=self.network_cmap, vmin=vmin, vmax=vmax)
            
            #re-set title after clear
            ax.set_title(title, fontsize=9, pad=5, color='white')
            
            #add shape information
            ax.set_xlabel(shape_label, fontsize=7, color='white')
            ax.tick_params(colors='white', labelsize=6)
            ax.set_yticks([])
            ax.set_xticks([])
        
        #redraw canvas
        self.viz_canvas.draw_idle()

#initialises and runs gui
def main():
    root = tk.Tk()
    app = EEGVisualisationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()