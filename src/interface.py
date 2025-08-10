import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from recognition import recognize_plate  # Assumindo que seu código principal está em recognize_plate.py

class PlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento de Placas Veiculares")
        
        # Configuração do layout
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variáveis
        self.image_path = None
        self.tk_image = None
        
        # Criar widgets
        self.create_widgets()
    
    def create_widgets(self):
        # Frame principal
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame para a imagem
        self.image_frame = tk.Frame(main_frame, bg='white', width=600, height=400)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.image_frame.pack_propagate(False)
        
        # Label para exibir a imagem
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Botão para carregar imagem
        load_btn = tk.Button(main_frame, text="Escolher Imagem", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botão para processar a imagem
        process_btn = tk.Button(main_frame, text="Processar Placa", command=self.process_plate)
        process_btn.pack(side=tk.LEFT)
        
        # Label para mostrar o resultado
        self.result_label = tk.Label(main_frame, text="", font=('Arial', 14))
        self.result_label.pack(side=tk.BOTTOM, pady=(10, 0))
    
    def load_image(self):
        """Abre uma janela para selecionar a imagem e a exibe na interface"""
        file_path = filedialog.askopenfilename(
            title="Selecione a imagem da placa",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp"), ("Todos os arquivos", "*.*")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.result_label.config(text="")
    
    def display_image(self, image_path):
        """Exibe a imagem selecionada na interface"""
        try:
            # Carregar imagem com OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Não foi possível carregar a imagem")
            
            # Converter de BGR para RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar mantendo a proporção
            img_pil = Image.fromarray(img)
            img_pil.thumbnail((600, 400))
            
            # Converter para formato Tkinter
            self.tk_image = ImageTk.PhotoImage(img_pil)
            
            # Atualizar label
            self.image_label.config(image=self.tk_image)
        
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar a imagem: {str(e)}")
    
    def process_plate(self):
        """Processa a imagem para reconhecer a placa"""
        if not self.image_path:
            messagebox.showwarning("Aviso", "Por favor, selecione uma imagem primeiro")
            return
        
        try:
            # Chamar sua função de reconhecimento
            plate_text, artifacts = recognize_plate(
                self.image_path, 
                "characters",  # Ajuste o caminho conforme necessário
                collect_artifacts=False
            )
            
            # Mostrar resultado
            if plate_text:
                self.result_label.config(text=f"Placa reconhecida: {plate_text}")
            else:
                self.result_label.config(text="Não foi possível reconhecer a placa")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar a placa: {str(e)}")
            self.result_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateRecognitionApp(root)
    root.mainloop()