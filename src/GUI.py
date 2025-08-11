import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from recognition import recognize_plate

class PlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento de Placas Veiculares")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Variáveis
        self.image_path = None
        self.artifacts = None
        self.plate_text = ""
        
        # Criar widgets
        self.create_widgets()
    
    def create_widgets(self):
        # Frame principal
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controles superiores
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(control_frame, text="Carregar Imagem", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Processar Placa", command=self.process_plate).pack(side=tk.LEFT, padx=5)
        self.result_label = tk.Label(control_frame, text="", font=('Arial', 14, 'bold'))
        self.result_label.pack(side=tk.LEFT, padx=10)
        
        # Notebook (abas) para as etapas do pipeline
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Abas
        self.tab_original = tk.Frame(self.notebook)
        self.tab_gray = tk.Frame(self.notebook)
        self.tab_binary = tk.Frame(self.notebook)
        self.tab_eroded = tk.Frame(self.notebook)
        self.tab_chars = tk.Frame(self.notebook)
        
        self.notebook.add(self.tab_original, text="Original")
        self.notebook.add(self.tab_gray, text="Escala de Cinza")
        self.notebook.add(self.tab_binary, text="Binária")
        self.notebook.add(self.tab_eroded, text="Erosão")
        self.notebook.add(self.tab_chars, text="Caracteres")
        
        # Labels para as imagens em cada aba
        self.img_labels = {
            'original': tk.Label(self.tab_original),
            'gray': tk.Label(self.tab_gray),
            'binary': tk.Label(self.tab_binary),
            'eroded': tk.Label(self.tab_eroded)
        }
        
        for label in self.img_labels.values():
            label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para os caracteres (usando grid)
        self.char_frames = []
        for i in range(7):  # 7 caracteres na placa mercosul
            frame = tk.Frame(self.tab_chars, bd=2, relief=tk.GROOVE)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            self.tab_chars.grid_columnconfigure(i, weight=1)
            
            label = tk.Label(frame)
            label.pack(fill=tk.BOTH, expand=True)
            
            char_label = tk.Label(frame, text=f"Char {i}", font=('Arial', 10))
            char_label.pack()
            
            self.char_frames.append({'frame': frame, 'img_label': label, 'text_label': char_label})
        
        self.tab_chars.grid_rowconfigure(0, weight=1)
    
    def load_image(self):
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
        if not self.image_path:
            messagebox.showwarning("Aviso", "Por favor, selecione uma imagem primeiro")
            return
        
        try:
            # Processar a placa e salvar os artefatos
            self.plate_text, self.artifacts = recognize_plate(
                self.image_path, 
                "characters",  # Ajuste o caminho conforme necessário
                collect_artifacts=True
            )
            
            # Mostrar resultado
            if self.plate_text:
                self.result_label.config(text=f"Placa reconhecida: {self.plate_text}")
                self.display_pipeline()
            else:
                self.result_label.config(text="Não foi possível reconhecer a placa")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar a placa: {str(e)}")
            self.result_label.config(text="")
    
    def display_pipeline(self):
        """Exibe todas as imagens do pipeline"""
        if not self.artifacts:
            return
        
        # Carregar e exibir cada etapa do processamento
        steps = [
            ('gray', '01_gray.png'),
            ('binary', '02_binary_inv_otsu.png'),
            ('eroded', '03_eroded.png')
        ]
        
        for step, filename in steps:
            img_path = os.path.join("out_artifacts", filename)
            if os.path.exists(img_path):
                self.display_step_image(step, img_path)
        
        # Carregar caracteres segmentados
        for i in range(7):  # 7 caracteres na placa mercosul
            char_path = os.path.join("out_artifacts", f"char_{i:02d}_std.png")
            if os.path.exists(char_path):
                self.display_char_image(i, char_path)
    
    def display_step_image(self, step, image_path):
        """Exibe uma imagem de uma etapa do processamento"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return
                
            # Converter para RGB se for colorida, ou repetir canal se for grayscale
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_pil = Image.fromarray(img)
            img_pil.thumbnail((800, 600))
            
            tk_img = ImageTk.PhotoImage(img_pil)
            self.img_labels[step].config(image=tk_img)
            self.img_labels[step].image = tk_img  # Manter referência
            
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
            frame_data['img_label'].image = tk_img  # Manter referência
            
            # Atualizar label com o caractere reconhecido (se disponível)
            if self.plate_text and char_index < len(self.plate_text):
                frame_data['text_label'].config(text=f"Char {char_index}: {self.plate_text[char_index]}")
            
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
            frame_data['text_label'].config(text=f"Char {self.char_frames.index(frame_data)}")

