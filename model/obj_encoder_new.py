import torch.nn as nn
import torch
from constants import ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN


class ObjectEncoder(nn.Module):
    def __init__(
        self, 
        type_encoder, 
        output_dim: int = 128,
        transformer_n_head: int = 8,
        transformer_dropout: float = 0.2,
        use_src_key_padding_mask: bool = False
    ) -> None:
        super().__init__()

        output_dim = int(output_dim)
        assert(
            output_dim % 2 == 0
        ),'output_dim or symbolic observation encoder should be divisible by 2'
        
        # feature_tensor = [type_embedding, parent_embedding, property, height, weight]

        # 20 is the observation length
        # batch_shape x observe_len -> batch_shape x observe_len x (output_dim / 2)

        # self.bert_tokenizer = bert_tokenizer
        # self.bert_model = bert_model
        # self.index2name = index2name

        if not torch.cuda.is_available():
            type_encoder = type_encoder.cpu()
        self.type_encoder = type_encoder

        # batch_shape x observe_len x (2 * output_dim) -> batch_shape x observe_len x output_dim
        self.state_encoder = nn.Sequential(         
            nn.Linear(in_features = 224, out_features = 224),
            nn.ReLU(),
            nn.Linear(in_features = 224, out_features = output_dim)
        )

        self.obj_embedding = nn.Parameter(data=torch.randn(1, output_dim))

        self.property_encoder = nn.Sequential(
            nn.Linear(in_features=6, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
        )

        self.height_encoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
        )

        self.weight_encoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(in_features=3, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
        )

        self.dis_encoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
        )

        self.use_src_key_padding_mask = use_src_key_padding_mask

        self.transformer_encoder_layer = nn.modules.TransformerEncoderLayer(
            d_model=output_dim, 
            nhead=transformer_n_head, 
            dropout=transformer_dropout, 
            dim_feedforward=output_dim,
            batch_first=True
        )

        self.transformer_encoder = nn.modules.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=4
        )

    def forward(
        self, 
        objs_type_id, 
        objs_parent_receptacle_id, 
        objs_properties, 
        objs_height, 
        objs_weight, 
        objs_pos, 
        objs_dis, 
        src_key_padding_mask,
    ):
        if torch.cuda.is_available():
            objs_properties = objs_properties.to(torch.float).cuda()
        else:
            objs_properties = objs_properties.to(torch.float)
            self.type_encoder = self.type_encoder.cpu()
        # print("for debug object encoder", objs_type_id.device, torch.cuda.is_available())

        objs_type_embedding = self.type_encoder(objs_type_id)
        objs_parent_receptacle_id_embedding = self.type_encoder(objs_parent_receptacle_id)

        # objs_temperature_embedding = self.temperature_encoder(objs_temperature)
        if torch.cuda.is_available():
            objs_height = objs_height.unsqueeze(-1).cuda()
            objs_weight = objs_weight.unsqueeze(-1).cuda()
            objs_dis = objs_dis.unsqueeze(-1).cuda()
        else:
            objs_height = objs_height.unsqueeze(-1)
            objs_weight = objs_weight.unsqueeze(-1)
            objs_dis = objs_dis.unsqueeze(-1)

        objs_properties_embedding = self.property_encoder(objs_properties)
        objs_height_embedding = self.height_encoder(objs_height.float())
        objs_weight_embedding = self.weight_encoder(objs_weight.float())
        objs_pos_embedding = self.pos_encoder(objs_pos.float())
        objs_dis_embedding = self.dis_encoder(objs_dis.float())

        # batch_shape x observe_len x (2 * output_dim)
        # print("for debug", objs_type_embedding.device, objs_parent_receptacle_id_embedding.device, objs_properties.device, objs_height.device, objs_weight.device)
        # print("for debug", objs_type_embedding.shape, objs_parent_receptacle_id_embedding.shape, objs_properties_embedding.shape, objs_height_embedding.shape, objs_weight_embedding.shape)
        _objs_state_embedding = torch.cat(   
            [
                objs_type_embedding,
                objs_parent_receptacle_id_embedding, 
                objs_properties_embedding, 
                objs_height_embedding, 
                objs_weight_embedding, 
                objs_pos_embedding, 
                objs_dis_embedding, 
            ],
            dim=-1
        )
        # batch_shape x observe_len x (2 * output_dim) -> batch_shape x observe_len x output_dim
        objs_state_embedding = self.state_encoder(_objs_state_embedding).unsqueeze(0)
        # print('objs_state_embedding.shape', objs_state_embedding.shape)
        # assert(
        #     len(objs_state_embedding.shape) == 4
        # ),'processed_observation_embedding has false dimension!!!'
        # batch_shape = 1 x batch_size
        batch_shape = objs_state_embedding.shape[0 : 3] 
        # if batch_shape[0] == 64:
        #     print('debug')
        # batch_shape x output_dim
        bs_obj_embedding = self.obj_embedding.repeat(*batch_shape, 1, 1)
        # 1 x batch_size x observe_len x output_dim -> 1 x batch_size x (observe_len+1) x output_dim
        objs_state_embedding = torch.cat(
            [
                bs_obj_embedding, 
                objs_state_embedding
            ],
            dim=-2
        )
        # embedding_shape = (observe_len+1) x output_dim
        embedding_shape = objs_state_embedding.shape[3 : ]  
        
        # 1 x batch_size x (observe_len+1) x output_dim -> batch_size x (observe_len+1) x output_dim
        objs_state_embedding_reshaped = objs_state_embedding.view(-1, *embedding_shape)
        
        # mask operation
        '''
        With respect to mask operation, the following variables are mentionable:
            - For BoolTensor src_key_padding_mask_bool: 
                - the positions with the value of True will be ignored,
                - while the position with the value of False will be unchanged.
            - For BoolTensor has_objs:
                - the positions with the value of True represent observed objects' number is not 0, 
                - while the positions with the value of False represent observed objects' number is 0. 
        
        The final src_key_padding_mask_bool = torch.eq(src_key_padding_mask_bool, has_objs).
        
        This is because when no object is observed, the values of original 
        src_key_padding_mask_bool are all True, leading to a 'nan' embedding. 
        Accordingly, the training process is interrupted because of the error. 
        
        In order to ensure that no errors are reported during the training process, 
        if no object is observed, the values of original src_key_padding_mask_bool 
        are all set to False.
        
        That is to say, if there are some observed objects, only these objects will
        be encoded, 'None' will not be encoded. If no object is observed, all the 'None'
        will be encoded to avoid error. 
        '''
        # src_key_padding_mask = src_key_padding_mask.unsqueeze(0)
        # mask_shape = observe_len
        mask_shape = src_key_padding_mask.shape[3 : ] 
        # 1 x batch_size x observe_len -> batch_size x observe_len
        if torch.cuda.is_available():
            src_key_padding_mask_bool = src_key_padding_mask.bool().cuda()
            has_objs = torch.sum(
                objs_type_id.view(-1, *objs_type_id.shape[3 : ]), dim=-1, keepdim=True
            ).bool().cuda()
        else:
            src_key_padding_mask_bool = src_key_padding_mask.bool()
            has_objs = torch.sum(
                objs_type_id.view(-1, *objs_type_id.shape[3 : ]), dim=-1, keepdim=True
            ).bool()
        # has_objs: 1 x batch_size x observe_len -> batch_size x 1
        # batch_size x observe_len -> batch_size x observe_len
        src_key_padding_mask_bool = torch.eq(src_key_padding_mask_bool, has_objs)

        obj_embedding_mask = torch.zeros(
            (src_key_padding_mask_bool.shape[0], objs_type_embedding.shape[1], 1), device=src_key_padding_mask_bool.device
        ).bool()
        # print('obj_embedding_mask.shape', obj_embedding_mask.shape, 'src_key_padding_mask_bool.shape', src_key_padding_mask_bool.shape)
        # print('obj_embedding_mask', obj_embedding_mask, 'src_key_padding_mask_bool', src_key_padding_mask_bool)
        src_key_padding_mask_bool = torch.cat(
            [
                obj_embedding_mask, 
                src_key_padding_mask_bool
            ],
            dim=-1
        )
        
        # batch_size x observe_len x output_dim -> 1 x batch_size x observe_len x output_dim
        if self.use_src_key_padding_mask == True:
            _obj_observation_embedding = self.transformer_encoder(
                objs_state_embedding_reshaped, src_key_padding_mask=src_key_padding_mask_bool
            ).view(*batch_shape, *embedding_shape)
        else:
            _obj_observation_embedding = self.transformer_encoder(
                objs_state_embedding_reshaped
            ).view(*batch_shape, *embedding_shape)
        
        # 1 x batch_size x observe_len x output_dim -> 1 x batch_size x output_dim
        obj_observation_embedding = _obj_observation_embedding[..., 0, :]

        return obj_observation_embedding
