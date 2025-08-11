import tkinter as tk  # Já vem com Python (para a interface gráfica)
from tkinter import ttk, filedialog, messagebox  # Componentes do Tkinter
from PIL import Image, ImageTk  # Para manipulação de imagens
import cv2  # OpenCV para processamento de imagens
import os  # Para operações com sistema de arquivos

from recognition import recognize_plate  # Seu módulo personalizado

class PlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento de Placas Veiculares")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Cores personalizadas
        self.button_color = "#285ffd"
        self.bg_color = "#101824"
        self.tab_selected = "#3d66b7"
        self.fg_color = "#ffffff"
        self.accent_color = "#3d66b7"
        
        self.root.configure(bg=self.bg_color)
        self.configure_styles()
        
        # Variáveis
        self.image_path = None
        self.artifacts = None
        self.plate_text = ""
        self.tk_image = None  # Adicionado para armazenar referência à imagem
        
        # Criar widgets
        self.create_widgets()
    
    def configure_styles(self):
        """Configura os estilos visuais da aplicação"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('.', 
                           background=self.bg_color,
                           foreground=self.fg_color)
        
        self.style.configure('TButton', 
                            background=self.button_color,
                            foreground=self.fg_color,
                            font=('Arial', 10, 'bold'),
                            borderwidth=0,
                            relief='flat',          
                            focuscolor=self.bg_color, 
                            focusthickness=0,
                            padding=5)
        
        self.style.map('TButton',
                    background=[('active', '#1a4fd1'), ('pressed', '#0e3cb1')],
                    relief=[('pressed', 'flat'), ('!pressed', 'flat')])
        
        self.style.configure('TNotebook', background=self.bg_color, focuscolor=self.bg_color)
        self.style.configure('TNotebook.Tab', 
                            background="#1a2232",
                            foreground=self.fg_color,
                            padding=[10, 5],
                            font=('Arial', 10, 'bold'))
        
        self.style.map('TNotebook.Tab',
                 focuscolor=[('selected', self.tab_selected)],  # Cor quando selecionado
                 background=[('selected', self.tab_selected)],
                 foreground=[('selected', 'white')])
    
    def create_widgets(self):
        """Cria todos os widgets da interface"""
        main_frame = tk.Frame(self.root, bg=self.bg_color, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = tk.Frame(main_frame, bg=self.bg_color)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_load = ttk.Button(control_frame, text="Carregar Imagem", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=5, pady=5)
        
        btn_process = ttk.Button(control_frame, text="Processar Placa", command=self.process_plate)
        btn_process.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.result_label = tk.Label(control_frame, 
                                   text="", 
                                   font=('Arial', 14, 'bold'),
                                   bg="#1a2232",
                                   fg="#ffffff",
                                   padx=10,
                                   pady=5,
                                   relief=tk.SUNKEN,
                                   bd=1)
        self.result_label.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        self.create_notebook(main_frame)
    
    def create_notebook(self, parent):
        """Cria o notebook com as abas de visualização"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Criar abas
        tabs = {
            'original': "Original",
            'gray': "Escala de Cinza",
            'binary': "Binária",
            'eroded': "Erosão",
            'chars': "Caracteres"
        }
        
        self.tabs = {}
        self.img_labels = {}
        
        for key, text in tabs.items():
            frame = tk.Frame(self.notebook, bg=self.bg_color)
            self.notebook.add(frame, text=text)
            self.tabs[key] = frame
            
            # Configuração diferente para a aba de caracteres
            if key == 'chars':
                # Remove qualquer widget existente
                for widget in frame.winfo_children():
                    widget.destroy()
                # Configura o grid para o frame principal
                frame.grid_rowconfigure(0, weight=1)
                for i in range(7):
                    frame.grid_columnconfigure(i, weight=1)
                self.create_char_frames(frame)
            else:
                label = tk.Label(frame, bg=self.bg_color)
                label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                self.img_labels[key] = label
    
    def create_char_frames(self, parent):
        """Cria os frames para exibição dos caracteres usando grid"""
        self.char_frames = []
        for i in range(7):  # 7 caracteres na placa mercosul
            frame = tk.Frame(parent, 
                            bg="#1a2232",
                            bd=2, 
                            relief=tk.GROOVE)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            
            # Frame interno para usar pack (dentro do frame que usa grid)
            inner_frame = tk.Frame(frame, bg="#1a2232")
            inner_frame.pack(fill=tk.BOTH, expand=True)
            
            label = tk.Label(inner_frame, bg="#1a2232")
            label.pack(fill=tk.BOTH, expand=True)
            
            char_label = tk.Label(inner_frame, 
                                text=f"Char {i}", 
                                font=('Arial', 10),
                                bg="#1a2232",
                                fg=self.fg_color)
            char_label.pack()
            
            self.char_frames.append({
                'frame': frame,
                'inner_frame': inner_frame,
                'img_label': label,
                'text_label': char_label
            })
    
    def load_image(self):
        """Carrega uma imagem para processamento"""
        file_path = filedialog.askopenfilename(
            title="Selecione a imagem da placa",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp"), ("Todos os arquivos", "*.*")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.result_label.config(text="")
            self.clear_pipeline()
    
    def display_image(self, image_path):
        """Exibe a imagem na interface"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Não foi possível carregar a imagem")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_pil.thumbnail((800, 600))
            
            self.tk_image = ImageTk.PhotoImage(img_pil)
            self.img_labels['original'].config(image=self.tk_image)
        
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar a imagem: {str(e)}")
    
    def process_plate(self):
        """Processa a imagem para reconhecimento da placa"""
        if not self.image_path:
            messagebox.showwarning("Aviso", "Por favor, selecione uma imagem primeiro")
            return
        
        try:
            self.result_label.config(text="Processando...", fg='white')
            self.root.update()
            
            self.plate_text, self.artifacts = recognize_plate(
                self.image_path, 
                "characters",
                collect_artifacts=True
            )
            
            if self.plate_text:
                self.result_label.config(
                    text=f"Placa reconhecida: {self.plate_text}",
                    fg=self.success_color
                )
                self.display_pipeline()
            else:
                self.result_label.config(
                    text="Não foi possível reconhecer a placa",
                    fg="#e74c3c"
                )
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar a placa: {str(e)}")
            self.result_label.config(text="", fg=self.fg_color)
    
    def display_pipeline(self):
        """Exibe todas as imagens do pipeline de processamento"""
        if not self.artifacts:
            return
        
        steps = [
            
            ('gray', '01_gray.png'),
            ('binary', '02_binary_inv_otsu.png'),
            ('eroded', '03_eroded.png')
        ]
        
        for step, filename in steps:
            img_path = os.path.join("out_artifacts", filename)
            if os.path.exists(img_path):
                self.display_step_image(step, img_path)
        
        for i in range(7):
            char_path = os.path.join("out_artifacts", f"char_{i:02d}_std.png")
            if os.path.exists(char_path):
                self.display_char_image(i, char_path)
    
    def display_step_image(self, step, image_path):
        """Exibe uma imagem de uma etapa do processamento"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return
                
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_pil = Image.fromarray(img)
            img_pil.thumbnail((800, 600))
            
            tk_img = ImageTk.PhotoImage(img_pil)
            self.img_labels[step].config(image=tk_img)
            self.img_labels[step].image = tk_img
            
        except Exception as e:
            print(f"Erro ao exibir imagem {step}: {str(e)}")
    
    def display_char_image(self, char_index, image_path):
        """Exibe um caractere segmentado"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return
                
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(img)
            img_pil.thumbnail((150, 150))
            
            tk_img = ImageTk.PhotoImage(img_pil)
            
            frame_data = self.char_frames[char_index]
            frame_data['img_label'].config(image=tk_img)
            frame_data['img_label'].image = tk_img
            
            if self.plate_text and char_index < len(self.plate_text):
                frame_data['text_label'].config(
                    text=f"Char {char_index}: {self.plate_text[char_index]}"
                )
            
        except Exception as e:
            print(f"Erro ao exibir caractere {char_index}: {str(e)}")
    
    def clear_pipeline(self):
        """Limpa todas as imagens do pipeline"""
        for label in self.img_labels.values():
            label.config(image='')
            label.image = None
        
        for frame_data in self.char_frames:
            frame_data['img_label'].config(image='')
            frame_data['img_label'].image = None
            frame_data['text_label'].config(
                text=f"Char {self.char_frames.index(frame_data)}"
            )

# Adicione esta definição de cor success_color se ainda não estiver no seu código
PlateRecognitionApp.success_color = "#2ecc71"  # Verde para sucesso