import torch
from torch import nn
from config import CFG
from loss_module import ArcMarginProduct
from transformers import AutoTokenizer, AutoModel

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class ShopeeBertModel(nn.Module):

    def __init__(
        self,
        n_classes = CFG.CLASSES_BERT,
        model_name = CFG.BERT_MODEL_NAME,
        fc_dim = CFG.FC_DIM_BERT,
        margin = CFG.MARGIN,
        scale = CFG.SCALE,
        use_fc = True
    ):

        super(ShopeeBertModel,self).__init__()

        print(f'Building Model Backbone for {model_name} model, margin = {CFG.MARGIN}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(CFG.DEVICE)

        in_features = 768
        self.use_fc = use_fc
        
        if use_fc:
            self.dropout = nn.Dropout(p=0.0)
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            in_features = fc_dim
            
        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            s = scale,
            m = margin,
            easy_margin = False,
            ls_eps = 0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, texts, labels=torch.tensor([0])):
        features = self.extract_features(texts)
        if self.training:
            logits = self.final(features, labels.to(CFG.DEVICE))
            return logits
        else:
            return features
        
    def extract_features(self, texts):
        encoding = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=CFG.MAX_LENGTH, return_tensors='pt').to(CFG.DEVICE)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        embedding = self.backbone(input_ids, attention_mask=attention_mask)
        x = mean_pooling(embedding, attention_mask)
        
        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)
        
        return x
