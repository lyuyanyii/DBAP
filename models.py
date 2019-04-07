import torch
import torch.nn as nn

class ProteinNet( nn.Module ):
    def __init__( self, proteinLen, embeddingDim ):
        super().__init__()

        self.embedding = nn.Embedding( proteinLen, embeddingDim )
        self.conv1 = nn.Sequential(
                nn.Conv1d( proteinLen, 32, 4 ),
                nn.ReLU(True),
                )
        self.conv2 = nn.Sequential(
                nn.Conv1d( 32, 64, 8 ),
                nn.ReLU(True),
                )
        self.conv3 = nn.Sequential(
                nn.Conv1d( 64, 128, 12 ),
                nn.ReLU(True),
                )

    def forward( self, x ):
        x = self.embedding( x )
        x = self.conv1( x )
        x = self.conv2( x )
        x = self.conv3( x )
        x = x.mean(2)
        return x

class CompoundNet( nn.Module ):
    def __init__( self, compoundLen, embeddingDim ):
        super().__init__()

        self.embedding = nn.Embedding( compoundLen, embeddingDim )
        self.conv1 = nn.Sequential(
                nn.Conv1d( compoundLen, 32, 4 ),
                nn.ReLU(True),
                )
        self.conv2 = nn.Sequential(
                nn.Conv1d( 32, 64, 6 ),
                nn.ReLU(True),
                )
        self.conv3 = nn.Sequential(
                nn.Conv1d( 64, 128, 8 ),
                nn.ReLU(True),
                )

    def forward( self, x ):
        x = self.embedding( x )
        x = self.conv1( x )
        x = self.conv2( x )
        x = self.conv3( x )
        x = x.mean(2)
        return x

class AffinityPredictNet( nn.Module ):
    def __init__( self, proteinLen, compoundLen, embeddingDim ):
        super().__init__()

        self.proteinNet = ProteinNet( proteinLen, embeddingDim )
        self.compoundNet = CompoundNet( compoundLen, embeddingDim )

        self.fc1 = nn.Sequential(
                nn.Dropout( p=0.1 ),
                nn.Linear( 256, 256 ),
                nn.ReLU(True),
                )
        self.fc2 = nn.Sequential(
                nn.Dropout( p=0.1 ),
                nn.Linear( 256, 128 ),
                nn.ReLU(True),
                )
        self.fc3 = nn.Sequential(
                nn.Dropout( p=0.1 ),
                nn.Linear( 128, 1 ),
                )

    def forward( self, protein, compound ):
        proteinFeature = self.proteinNet( protein )
        compoundFeature = self.compoundNet( compound )
        feature = torch.cat( (proteinFeature, compoundFeature), 1 )

        x = self.fc1( feature )
        x = self.fc2( x )
        x = self.fc3( x )

        return x
