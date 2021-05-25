import timm
from torch import nn
from config import CFG
from loss_module import ArcMarginProduct, CurricularFace

class ShopeeModel(nn.Module):

    def __init__(
        self,
        n_classes = CFG.CLASSES,
        model_name = CFG.MODEL_NAME,
        fc_dim = CFG.FC_DIM,
        margin = CFG.MARGIN,
        scale = CFG.SCALE,
        use_fc = True,
        pretrained = True,
        use_arcface = CFG.USE_ARCFACE):

        super(ShopeeModel,self).__init__()
        print(f'Building Model Backbone for {model_name} model, margin = {margin}')

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if 'efficientnet' in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        
        elif 'resnet' in model_name:
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif 'resnext' in model_name:
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif 'densenet' in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif 'nfnet' in model_name:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.pooling =  nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.0)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        if use_arcface:
            self.final = ArcMarginProduct(final_in_features, 
                                                n_classes, 
                                                s=scale, 
                                                m=margin)
        else:
            self.final = CurricularFace(final_in_features, 
                                                n_classes, 
                                                s=scale, 
                                                m=margin)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        feature = self.extract_feat(image)
        logits = self.final(feature,label)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x
