from flask import Flask, render_template, request
import numpy as np
import torch

app = Flask(__name__)

# Vos données d'entraînement
x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4, 1.5]), dtype=float)
y = np.array(([1], [0], [1], [0], [1], [0], [1], [0], [1]), dtype=float)  # Ajout d'une étiquette supplémentaire

# Convertir les données en tensors PyTorch
X = torch.tensor(x_entrer, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Normalisation des données
X = X / torch.max(X, dim=0).values

# Définir votre classe de réseau neuronal avec PyTorch
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(2, 4)  # Une seule couche cachée avec 4 neurones
        self.layer2 = torch.nn.Linear(4, 1)  # Couche de sortie

    def forward(self, X):
        z = torch.relu(self.layer1(X))  # Fonction d'activation ReLU pour la couche cachée
        o = torch.sigmoid(self.layer2(z))  # Fonction d'activation sigmoid pour la couche de sortie
        return o

# Instancier le modèle, la fonction de perte et l'optimiseur
model = NeuralNetwork()
criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss pour un problème de classification binaire
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Entraînement du modèle
for epoch in range(6000):
    # Forward pass
    output = model(X)
    
    # Calcul de la perte
    loss = criterion(output, y)
    
    # Rétropropagation et mise à jour des poids
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f'Epoch {epoch}/{6000} - Loss: {loss.item()}')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        length = float(request.form['length'])
        width = float(request.form['width'])

        with torch.no_grad():
            user_input = torch.tensor([[length, width]], dtype=torch.float32)
            user_input = user_input / torch.max(X, dim=0).values  # Normaliser les données
            prediction = model(user_input)

        color_prediction = "Rouge" if prediction.item() > 0.6 else "Bleu"
        color_class = "red" if prediction.item() > 0.6 else "blue"

        return render_template('result.html', length=length, width=width, prediction=color_prediction, color_class=color_class)

if __name__ == '__main__':
    app.run(debug=True)
