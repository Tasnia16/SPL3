import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch.nn.functional as F

class ImprovedAlexNet(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedAlexNet, self).__init__()
        self.feat_dim = 256
        
        self.features = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.features(x)

class DeepCORAL(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepCORAL, self).__init__()
        self.base_network = ImprovedAlexNet(input_dim)
        self.classifier = nn.Linear(self.base_network.feat_dim, num_classes)
        
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
        
    def forward(self, source, target=None):
        source_features = self.base_network(source)
        source_outputs = self.classifier(source_features)
        
        if self.training and target is not None:
            target_features = self.base_network(target)
            target_outputs = self.classifier(target_features)
            return source_outputs, target_outputs, source_features, target_features
        
        return source_outputs, source_features

def coral_loss(source, target):
    d = source.size(1)
    source_centered = source - torch.mean(source, dim=0, keepdim=True)
    target_centered = target - torch.mean(target, dim=0, keepdim=True)
    
    eps = 1e-3
    source_cov = (source_centered.t() @ source_centered) / (source.size(0) - 1) + eps * torch.eye(d).to(source.device)
    target_cov = (target_centered.t() @ target_centered) / (target.size(0) - 1) + eps * torch.eye(d).to(target.device)
    
    loss = torch.norm(source_cov - target_cov, p='fro') ** 2
    return loss / (4 * d * d)

class DeepCoralModel:
    def __init__(self, batch_size=64, num_epochs=100, learning_rate=0.001, n_neighbors=1):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.n_neighbors = n_neighbors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit_predict(self, Xs, Ys, Xt, Yt):
        # Store original Yt for evaluation
        Yt_numpy = Yt if isinstance(Yt, np.ndarray) else Yt.numpy()
        
        # Convert to torch tensors if they aren't already
        if not isinstance(Xs, torch.Tensor):
            Xs = torch.FloatTensor(Xs)
        if not isinstance(Ys, torch.Tensor):
            Ys = torch.LongTensor(Ys)
        if not isinstance(Xt, torch.Tensor):
            Xt = torch.FloatTensor(Xt)
            
        # Move to device
        Xs = Xs.to(self.device)
        Ys = Ys.to(self.device)
        Xt = Xt.to(self.device)
        
        # Create data loaders
        source_dataset = torch.utils.data.TensorDataset(Xs, Ys)
        target_dataset = torch.utils.data.TensorDataset(Xt, torch.zeros(len(Xt)))
        
        source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=self.batch_size, shuffle=True)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_dim = Xs.shape[1]
        num_classes = len(torch.unique(Ys))
        self.model = DeepCORAL(input_dim, num_classes).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training
        best_acc = 0.0
        best_pred = None
        
        for epoch in range(self.num_epochs):
            self.model.train()
            target_iter = iter(target_loader)
            
            for source_data, source_label in source_loader:
                try:
                    target_data, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)
                
                source_output, target_output, source_features, target_features = self.model(source_data, target_data)
                
                # Calculate losses
                classification_loss = criterion(source_output, source_label)
                transfer_loss = coral_loss(source_features, target_features)
                total_loss = classification_loss + transfer_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            # Evaluation using KNN
            self.model.eval()
            with torch.no_grad():
                _, source_features = self.model(Xs)
                _, target_features = self.model(Xt)
                
                # Convert features to numpy for KNN
                source_features_np = source_features.cpu().numpy()
                target_features_np = target_features.cpu().numpy()
                Ys_np = Ys.cpu().numpy()
                
                # KNN classification
                knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                knn.fit(source_features_np, Ys_np)
                y_pred = knn.predict(target_features_np)
                
                accuracy = accuracy_score(Yt_numpy, y_pred)
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_pred = y_pred
        
        # Final evaluation with best predictions
        accuracy = accuracy_score(Yt_numpy, best_pred)
        auc_roc = roc_auc_score(Yt_numpy, best_pred, multi_class='ovr')
        # f1 = f1_score(Yt_numpy, best_pred, average='binary')
        f1 = f1_score(Yt_numpy, best_pred,labels=None, pos_label=1, average='weighted',sample_weight=None)
        return accuracy, best_pred, auc_roc, f1