import torch

def load_k_v_queryR(input_ent, ent_to_neighbors, ent_to_relations, ent_to_outORin, embed_ent, r_embed, device='cpu', dk_layers=2):
        """
        Get k and v vectors for `CokeBert`.

        Args:
            input_ent: input_ent: a torch.LongTensor of shape [batch_size, sequence_length,embedding_size]
                    with the entities embeddings
            ent_to_neighbors: map from entity to neighbors in Knowledge Graph (loaded from `e1_e2_list_2D_Tensor.pkl`)
            ent_to_relations: map from entity to neighboring relations in Knowledge Graph (loaded from `e1_r_list_2D_Tensor.pkl`)
            ent_to_outORin: direction of arc in knowledge graph (1: out of ent, -1: into ent) (loaded from `e1_outORin_list_2D_Tensor.pkl`)
            embed_ent: Map of entity embeddings
            r_embed: Map of relation embeddings
            device: The device to load the resulting vectors onto (default: `cpu`) 
            dk_layers (int): The number of layers in the dynamic knowledge encoder, therefore also half the number of output vectors (default: 2)

        Returns:
            list of k_i, v_i vectors as input for the dynamic knowledge encoder
        """"
        input_ent = input_ent.cpu()

        ent_pos_s = torch.nonzero(input_ent)

        max_entity=0
        value=0
        idx_1 = 0
        last_part = 0
        for idx_2,x in enumerate(ent_pos_s):
            if int(x[0]) != value:
                max_entity = max(idx_2-idx_1,max_entity)
                idx_1 = idx_2
                value = int(x[0])
                last_part = 1
            else:
                last_part+=1
        max_entity = max(last_part,max_entity)

        new_input_ent = list()
        for i_th, ten in enumerate(input_ent):
            ten_ent = ten[ten!=0]
            new_input_ent.append( torch.cat( (ten_ent,( torch.LongTensor( [0]*(max_entity-ten_ent.shape[0]) ) ) ) ) )

        input_ent = torch.stack(new_input_ent)


        k_v_s = []
        input_ent_neighbor = input_ent

        for i in range(dk_layers):
            #Neighbor
            input_ent_neighbor_i = torch.index_select(ent_to_neighbors,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()

            #create input_ent_neighbor_i
            input_ent_neighbor_emb_i = torch.index_select(embed_ent,0,input_ent_neighbor_i.reshape(input_ent_neighbor_i.shape[0]*input_ent_neighbor_i.shape[1])) #
            input_ent_neighbor_emb_i = input_ent_neighbor_emb_i.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_neighbors.shape[1],embed_ent.shape[-1])

            #create input_ent_r_1:
            input_ent_r_emb_i = torch.index_select(ent_to_relations,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()
            input_ent_r_emb_i = torch.index_select(r_embed,0,input_ent_r_emb_i.reshape(input_ent_r_emb_i.shape[0]*input_ent_r_emb_i.shape[1])) #
            input_ent_r_emb_1 = input_ent_r_emb_i.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_relations.shape[1],r_embed.shape[-1])

            #create outORin_1:
            input_ent_outORin_emb_i = torch.index_select(ent_to_outORin,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1]))
            input_ent_outORin_emb_i = input_ent_outORin_emb_i.reshape(input_ent.shape[0],input_ent.shape[1],input_ent_outORin_emb_i.shape[1])
            input_ent_outORin_emb_i = input_ent_outORin_emb_i.unsqueeze(i+3)

            k_i= input_ent_outORin_emb_i.to(device=device)*input_ent_r_emb_i.to(device=device)
            v_i = input_ent_neighbor_emb_i.to(device=device)+k_1

            input_ent_neighbor = input_ent_neighbor_i

            k_v_s.append(k_i)
            k_v_s.append(v_i)

        """
        #Neighbor
        input_ent_neighbor = torch.index_select(ent_to_neighbors,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()

        #create input_ent_neighbor_1

        input_ent_neighbor_emb_1 = torch.index_select(embed_ent,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])) #
        input_ent_neighbor_emb_1 = input_ent_neighbor_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_neighbors.shape[1],embed_ent.shape[-1])

        #create input_ent_r_1:
        input_ent_r_emb_1 = torch.index_select(ent_to_relations,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
        input_ent_r_emb_1 = torch.index_select(r_embed,0,input_ent_r_emb_1.reshape(input_ent_r_emb_1.shape[0]*input_ent_r_emb_1.shape[1])) #
        input_ent_r_emb_1 = input_ent_r_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_relations.shape[1],r_embed.shape[-1])

        #create outORin_1:
        input_ent_outORin_emb_1 = torch.index_select(ent_to_outORin,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1]))
        input_ent_outORin_emb_1 = input_ent_outORin_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],input_ent_outORin_emb_1.shape[1])
        input_ent_outORin_emb_1 = input_ent_outORin_emb_1.unsqueeze(3)


        #create input_ent_neighbor_2
        input_ent_neighbor_2 = torch.index_select(ent_to_neighbors,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()

        input_ent_neighbor_emb_2 = torch.index_select(embed_ent,0,input_ent_neighbor_2.reshape(input_ent_neighbor_2.shape[0]*input_ent_neighbor_2.shape[1])) #
        input_ent_neighbor_emb_2 = input_ent_neighbor_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_neighbors.shape[1],ent_to_neighbors.shape[1],embed_ent.shape[-1])


        #create input_ent_r_2:
        input_ent_r_2 = torch.index_select(ent_to_relations,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()

        input_ent_r_emb_2 = torch.index_select(r_embed,0,input_ent_r_2.reshape(input_ent_r_2.shape[0]*input_ent_r_2.shape[1])) #
        input_ent_r_emb_2 = input_ent_r_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_to_relations.shape[1],ent_to_neighbors.shape[1],r_embed.shape[-1])

        #create outORin_2: #?
        input_ent_outORin_emb_2 = torch.index_select(ent_to_outORin,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1]))

        input_ent_outORin_emb_2 = input_ent_outORin_emb_2.reshape(input_ent_r_emb_2.shape[0],input_ent_r_emb_2.shape[1],input_ent_r_emb_2.shape[2],input_ent_r_emb_2.shape[3])
        input_ent_outORin_emb_2 = input_ent_outORin_emb_2.unsqueeze(4)

        k_1 = input_ent_outORin_emb_1.to(device=device)*input_ent_r_emb_1.to(device=device)
        v_1 = input_ent_neighbor_emb_1.to(device=device)+k_1
        k_2 = input_ent_outORin_emb_2.to(device=device)*input_ent_r_emb_2.to(device=device)
        v_2 = input_ent_neighbor_emb_2.to(device=device)+k_2
        """

        #return k_1,v_1,k_2,v_2
        return k_v_s