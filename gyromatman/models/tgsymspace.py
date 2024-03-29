import torch
from geoopt.linalg import batch_linalg as lalg
from gyromatman.config import INIT_EPS, DEVICE
from gyromatman.models.base import KGModel
from gyromatman.utils import productory, trace, skew_partial, tril
from gyromatman.manifolds import SPDLEManifold, GRManifold
from gyromatman.manifolds.metrics import MetricType

__all__ = ["TgSPDLEModel", "TgGRModel", "TgSPDLEGRModel", "TgGyroGRModel"]

class TgGRModel(KGModel):
    """Knowledge Graph embedding model that operates on the tangent space of the Grassmann Manifold"""

    def __init__(self, args):
        super().__init__(args)
        self.manifold = GRManifold(dims=args.dims, pdim=args.pdim, metric=MetricType.from_str(args.metric))

        init_fn = lambda n_points: torch.randn((n_points, args.pdim, args.dims - args.pdim)) * INIT_EPS
        self.entities = torch.nn.Parameter(init_fn(args.num_entities), requires_grad=True)    # num_entities  x p x (n-p)
        self.relations = torch.nn.Parameter(init_fn(args.num_relations), requires_grad=True)  # num_relations x p x (n-p)
        # rel_transforms are n x n symmetric matrices
        self.rel_transforms = torch.nn.Parameter(                                             # num_relations x p x (n-p)                
            torch.rand((args.num_relations, args.pdim, args.dims - args.pdim)) * 2 - 1.0,     # U[-1, 1]
            requires_grad=True
        )

        self.addition = self.addition_hrh if args.use_hrh == 1 else self.addition_rhr
        #if args.inverse_tail == 1:
            # performs exponential map and inverse in only one diagonalization
        #    self.map_tail = lambda tg_tails: lalg.sym_funcm(tg_tails, lambda tensor: torch.reciprocal(torch.exp(tensor)))
        #else:
        #    self.map_tail = self.manifold.expmap_id
        self.map_tail = self.manifold.expmap_id

    def get_lhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n x n
        """
        tg_heads = self.entities[triples[:, 0]]                                
        rel_transforms = self.rel_transforms[triples[:, 1]]                     

        tg_heads = skew_partial(rel_transforms * tg_heads)                     
        tg_relations = skew_partial(self.relations[triples[:, 1]])                  

        return self.addition(tg_heads, tg_relations)                            

    def addition_hrh(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaGrassmann  (Skew-symmetric matrices)
        :param relations: b x n x n: Points in TgSpaGrassmann (Skew-symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """         
        return self.manifold.addition_id_from_skew(entities, relations)           

    def addition_rhr(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaGrassmann (Skew-symmetric matrices)
        :param relations: b x n x n: Points in TgSpaGrassmann (Skew-symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """
        return self.addition_hrh(relations, entities)

    def get_rhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n
        """
        tg_tails = skew_partial(self.entities[triples[:, 2]])                  
        return self.map_tail(tg_tails)

    def similarity_score(self, lhs, rhs):
        dist, _ = self.manifold.dist(lhs, rhs)
        return -1 * dist ** 2, dist

    def similarity_score_eval(self, lhs, rhs):
        dist, _ = self.manifold.dist_eval(lhs, rhs)
        return -1 * dist ** 2, dist

    def get_factors(self, triples):
        """
        Returns factors for embeddings' regularization.
        :param triples: b x 3: (head, relation, tail)
        :return: list of 3 tensors of b x *
        """
        heads = self.entities[triples[:, 0]]
        rel = self.relations[triples[:, 1]]
        rel_transf = self.rel_transforms[triples[:, 1]]
        tails = self.entities[triples[:, 2]]
        return heads, rel, rel_transf, tails

    def compute_norms(self, points):
        entities = self.manifold.expmap_id(lalg.sym(points.detach()))
        return entities.flatten(start_dim=1).norm(dim=-1)

    def entity_norms(self):
        return self.compute_norms(self.entities)

    def relation_norms(self):
        return self.compute_norms(self.relations)

    def relation_transform_norms(self):
        return self.rel_transforms.detach().flatten(start_dim=1).norm(dim=-1)


class TgSPDLEModel(KGModel):
    """Knowledge Graph embedding model that operates on the tangent space of the SPD Manifold with Log-Euclidean metrics"""

    def __init__(self, args):
        super().__init__(args)
        self.manifold = SPDLEManifold(dims=args.dims, metric=MetricType.from_str(args.metric))

        init_fn = lambda n_points: torch.randn((n_points, args.dims, args.dims)) * INIT_EPS
        self.entities = torch.nn.Parameter(init_fn(args.num_entities), requires_grad=True)    
        self.relations = torch.nn.Parameter(init_fn(args.num_relations), requires_grad=True)  
        # rel_transforms are n x n symmetric matrices
        self.rel_transforms = torch.nn.Parameter(                                                            
            torch.rand((args.num_relations, args.dims, args.dims)) * 2 - 1.0,  # U[-1, 1]
            requires_grad=True
        )

        self.addition = self.addition_hrh if args.use_hrh == 1 else self.addition_rhr
        if args.inverse_tail == 1:
            # performs exponential map and inverse in only one diagonalization
            self.map_tail = lambda tg_tails: lalg.sym_funcm(tg_tails, lambda tensor: torch.reciprocal(torch.exp(tensor)))
        else:
            self.map_tail = self.manifold.expmap_id

    def get_lhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n x n
        """
        tg_heads = lalg.sym(self.entities[triples[:, 0]])                   
        rel_transforms = lalg.sym(self.rel_transforms[triples[:, 1]])       

        tg_heads = rel_transforms * tg_heads                                
        tg_relations = lalg.sym(self.relations[triples[:, 1]])              

        return self.addition(tg_heads, tg_relations)

    def addition_hrh(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :param relations: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """        
        return SPDLEManifold.addition_id_from_log(entities, relations)           

    def addition_rhr(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :param relations: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """
        return self.addition_hrh(relations, entities)

    def get_rhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n
        """
        tg_tails = lalg.sym(self.entities[triples[:, 2]])                  
        return tg_tails

    def similarity_score(self, lhs, rhs):
        dist, _ = self.manifold.dist(lhs, rhs)
        return -1 * dist ** 2, dist

    def similarity_score_eval(self, lhs, rhs):
        dist, _ = self.manifold.dist_eval(lhs, rhs)
        return -1 * dist ** 2, dist

    def get_factors(self, triples):
        """
        Returns factors for embeddings' regularization.
        :param triples: b x 3: (head, relation, tail)
        :return: list of 3 tensors of b x *
        """
        heads = self.entities[triples[:, 0]]
        rel = self.relations[triples[:, 1]]
        rel_transf = self.rel_transforms[triples[:, 1]]
        tails = self.entities[triples[:, 2]]
        return heads, rel, rel_transf, tails

    def compute_norms(self, points):
        entities = self.manifold.expmap_id(lalg.sym(points.detach()))
        return entities.flatten(start_dim=1).norm(dim=-1)

    def entity_norms(self):
        return self.compute_norms(self.entities)

    def relation_norms(self):
        return self.compute_norms(self.relations)

    def relation_transform_norms(self):
        return self.rel_transforms.detach().flatten(start_dim=1).norm(dim=-1)


class TgSPDLEGRModel(KGModel):
    """Knowledge Graph embedding model that operates on the tangent space of the SPD Manifold with Log-Euclidean metrics"""

    def __init__(self, args):
        super().__init__(args)

        self.manifold_spd = SPDLEManifold(dims=args.pdim, metric=MetricType.from_str(args.metric))

        init_fn_spd = lambda n_points: torch.randn((n_points, args.pdim, args.pdim)) * INIT_EPS
        self.entities_spd = torch.nn.Parameter(init_fn_spd(args.num_entities), requires_grad=True)    
        self.relations_spd = torch.nn.Parameter(init_fn_spd(args.num_relations), requires_grad=True)  
        # rel_transforms are n x n symmetric matrices
        self.rel_transforms_spd = torch.nn.Parameter(                                                           
            torch.rand((args.num_relations, args.pdim, args.pdim)) * 2 - 1.0,  # U[-1, 1]
            requires_grad=True
        )

        self.addition_spd = self.addition_hrh_spd if args.use_hrh_spd == 1 else self.addition_rhr_spd
        if args.inverse_tail == 1:
            # performs exponential map and inverse in only one diagonalization
            self.map_tail_spd = lambda tg_tails: lalg.sym_funcm(tg_tails, lambda tensor: torch.reciprocal(torch.exp(tensor)))
        else:
            self.map_tail_spd = self.manifold_spd.expmap_id

        self.manifold_gr = GRManifold(dims=args.dims, pdim=args.kdim, metric=MetricType.from_str(args.metric))

        init_fn_gr = lambda n_points: torch.randn((n_points, args.kdim, args.dims - args.kdim)) * INIT_EPS        
        self.entities_gr = torch.nn.Parameter(init_fn_gr(args.num_entities), requires_grad=True)    
        self.relations_gr = torch.nn.Parameter(init_fn_gr(args.num_relations), requires_grad=True)  
        # rel_transforms are n x n symmetric matrices
        self.rel_transforms_gr = torch.nn.Parameter(                                                     
            torch.rand((args.num_relations, args.kdim, args.dims - args.kdim)) * 2 - 1.0,     # U[-1, 1]            
            requires_grad=True
        )

        self.addition_gr = self.addition_hrh_gr if args.use_hrh_gr == 1 else self.addition_rhr_gr
        self.map_tail_gr = self.manifold_gr.expmap_id

        self.dist_factor = args.dist_factor

    def get_lhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n x n
        """
        tg_heads_spd = lalg.sym(self.entities_spd[triples[:, 0]])                  
        rel_transforms_spd = lalg.sym(self.rel_transforms_spd[triples[:, 1]])       

        tg_heads_spd = rel_transforms_spd * tg_heads_spd                            
        tg_relations_spd = lalg.sym(self.relations_spd[triples[:, 1]])              

        tg_heads_gr = self.entities_gr[triples[:, 0]]                                     
        rel_transforms_gr = self.rel_transforms_gr[triples[:, 1]]                         

        tg_heads_gr = skew_partial(rel_transforms_gr * tg_heads_gr)                             
        tg_relations_gr = skew_partial(self.relations_gr[triples[:, 1]])                  

        return self.addition_spd(tg_heads_spd, tg_relations_spd), self.addition_gr(tg_heads_gr, tg_relations_gr)

    def addition_hrh_spd(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :param relations: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """        
        return SPDLEManifold.addition_id_from_log(entities, relations)          

    def addition_rhr_spd(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :param relations: b x n x n: Points in TgSpaSPD (Symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """
        # inverts the order of the addition
        return self.addition_hrh_spd(relations, entities)

    def addition_hrh_gr(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaGrassmann  (Skew-symmetric matrices)
        :param relations: b x n x n: Points in TgSpaGrassmann (Skew-symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """       
        return self.manifold_gr.addition_id_from_skew(entities, relations)           

    def addition_rhr_gr(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaGrassmann (Skew-symmetric matrices)
        :param relations: b x n x n: Points in TgSpaGrassmann (Skew-symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """
        # inverts the order of the addition
        return self.addition_hrh_gr(relations, entities)

    def get_rhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n
        """
        tg_tails_spd = lalg.sym(self.entities_spd[triples[:, 2]])                  

        tg_tails_gr = skew_partial(self.entities_gr[triples[:, 2]])                  

        return tg_tails_spd, self.map_tail_gr(tg_tails_gr)

    def similarity_score(self, lhs, rhs):
        lhs_spd, lhs_gr = lhs
        rhs_spd, rhs_gr = rhs
        dist_spd, _ = self.manifold_spd.dist(lhs_spd, rhs_spd)
        dist_gr, _ = self.manifold_gr.dist(lhs_gr, rhs_gr)        
        dist = self.dist_factor * dist_spd + dist_gr
        return -1 * dist ** 2, dist

    def similarity_score_eval(self, lhs, rhs):
        lhs_spd, lhs_gr = lhs
        rhs_spd, rhs_gr = rhs
        dist_spd, _ = self.manifold_spd.dist_eval(lhs_spd, rhs_spd)
        dist_gr, _ = self.manifold_gr.dist_eval(lhs_gr, rhs_gr)
        dist = self.dist_factor * dist_spd + dist_gr
        return -1 * dist ** 2, dist

    def get_factors(self, triples):
        """
        Returns factors for embeddings' regularization.
        :param triples: b x 3: (head, relation, tail)
        :return: list of 3 tensors of b x *
        """
        heads_spd = self.entities_spd[triples[:, 0]]
        rel_spd = self.relations_spd[triples[:, 1]]
        rel_transf_spd = self.rel_transforms_spd[triples[:, 1]]
        tails_spd = self.entities_spd[triples[:, 2]]

        heads_gr = self.entities_gr[triples[:, 0]]
        rel_gr = self.relations_gr[triples[:, 1]]
        rel_transf_gr = self.rel_transforms_gr[triples[:, 1]]
        tails_gr = self.entities_gr[triples[:, 2]]

        return heads_spd, rel_spd, rel_transf_spd, tails_spd, heads_gr, rel_gr, rel_transf_gr, tails_gr

    def compute_norms(self, points):
        entities = self.manifold_spd.expmap_id(lalg.sym(points.detach()))
        return entities.flatten(start_dim=1).norm(dim=-1)

    def entity_norms(self):
        return self.compute_norms(self.entities_spd)

    def relation_norms(self):
        return self.compute_norms(self.relations_spd)

    def relation_transform_norms(self):
        return self.rel_transforms_spd.detach().flatten(start_dim=1).norm(dim=-1)
    
    
class TgGyroGRModel(KGModel):
    """Knowledge Graph embedding isometry model that operates on the tangent space of the Grassmann Manifold"""

    def __init__(self, args):
        super().__init__(args)

        self.dim_pp = args.pdim
        self.dim_np = args.dims - args.pdim
        self.dim_nn = self.dim_pp + self.dim_np

        self.manifold = GRManifold(dims=args.dims, pdim=args.pdim, metric=MetricType.from_str(args.metric))

        init_fn = lambda n_points: torch.randn((n_points, self.dim_pp, self.dim_np)) * INIT_EPS        
        self.entities = torch.nn.Parameter(init_fn(args.num_entities), requires_grad=True)    # num_entities  x p x (n-p)
        self.relations = torch.nn.Parameter(init_fn(args.num_relations), requires_grad=True)  # num_relations x p x (n-p)                

        # extended version 
        self.isom_init_nn = lambda n: torch.rand((n, self.dim_nn * (self.dim_nn - 1) // 2)) * 0.5 - 0.25  # U[-0.25, 0.25] radians ~ U[-15°, 15°]
        self.rot_params_nn = torch.nn.Parameter(self.isom_init_nn(args.num_relations), requires_grad=True)
        self.embed_index_nn = self.get_isometry_embed_index(self.dim_nn)        

        self.addition = self.addition_hrh if args.use_hrh == 1 else self.addition_rhr
        
        self.map_tail = self.manifold.expmap_id
    
    def get_isometry_embed_index(self, dims):        
        # indexes := 1 <= i < j < n. Using 1-based notation to make it equivalent to matrix math notation
        indexes = [(i, j) for i in range(1, dims + 1) for j in range(i + 1, dims + 1)]

        embed_index = []
        for i, j in indexes:
            row = []
            for c_i, c_j in [(i, i), (i, j), (j, i), (j, j)]:  # 4 combinations that we care for each (i, j) pair
                flatten_index = dims * (c_i - 1) + c_j - 1
                row.append(flatten_index)
            embed_index.append(row)
        return torch.LongTensor(embed_index).unsqueeze(0).to(DEVICE)  # 1 x m x 4

    def get_lhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n x n
        """               
        # extended version
        isometry_params_nn = self.get_isometry_params(self.rot_params_nn)
        all_relation_isometries = self.build_relation_isometry_matrices(isometry_params_nn, self.embed_index_nn, self.dim_nn)    # r x n x n
        rel_isometries = all_relation_isometries[triples[:, 1]]                             # b x n x n

        # method 1
        tg_heads = skew_partial(self.entities[triples[:, 0]])
        tg_heads = self.manifold.isometry_map(rel_isometries, tg_heads)
        tg_relations = skew_partial(self.relations[triples[:, 1]])        

        return self.addition(tg_heads, tg_relations)

    def get_isometry_params(self, isom_params):
        """
        This method must be implemented by concrete clases where the isometry parameters
        for each relation are computed  and returned as a tensor of r x m x 4 where
            r: num of relations
            m: num of isometries
        :return: tensor of r x m x 4
        """
        return self.compute_reflection_params(isom_params)

    def compute_reflection_params(self, params):
        """
        Computes rotation parameters:
        For each entry in params computes:
            R^+ = (cos(x), -sin(x), sin(x), cos(x))
        :param params: r x m
        :return: r x m x 4
        """
        cos_x = torch.cos(params)
        sin_x = torch.sin(params)        
        res = torch.stack([cos_x, sin_x, sin_x, -cos_x], dim=-1)
        return res

    def build_relation_isometry_matrices(self, isom_params, embed_index, dims):
        """
        Builds the rotation isometries as matrices for all available relations
        :param isom_params: r x m x 4
        :return: r x n x n
        """
        # isom_params = self.compute_rotation_params(self.rot_params)  # r x m x 4
        embeded_rotations = self.embed_params(isom_params, embed_index, dims)  # r x m x n x n
        isom_rot = productory(embeded_rotations)  # r x n x n
        return isom_rot

    def embed_params(self, iso_params: torch.Tensor, embed_index: torch.Tensor, dims: int) -> torch.Tensor:
        """
        Embeds the isometry params.
        For each isometric operation there are m isometries with 4 params each.
        This method embeds the 4 params into a dims x dims identity, in positions given by self.embed_index

        :param iso_params: b x m x 4, where m = dims * (dims - 1) / 2, which is the amount of isometries
        :param dims: (also called n) dimension of output identities, with params embedded
        :return: b x m x n x n
        """
        bs, m, _ = iso_params.size()
        target = torch.eye(dims, requires_grad=True, device=iso_params.device)
        target = target.reshape(1, 1, dims * dims).repeat(bs, m, 1)  # b x m x n * n
        scatter_index = embed_index.repeat(bs, 1, 1)  # b x m x 4
        embed_isometries = target.scatter(dim=-1, index=scatter_index, src=iso_params)  # b x m x n * n
        embed_isometries = embed_isometries.reshape(bs, m, dims, dims)  # b x m x n x n
        return embed_isometries
    
    def addition_hrh(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaGrassmann  (Skew-symmetric matrices)
        :param relations: b x n x n: Points in TgSpaGrassmann (Skew-symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """        
        # method 1
        return self.manifold.addition_rot_and_skew(entities, relations)                   

    def addition_rhr(self, entities, relations):
        """
        :param entities: b x n x n: Points in TgSpaGrassmann (Skew-symmetric matrices)
        :param relations: b x n x n: Points in TgSpaGrassmann (Skew-symmetric matrices)
        :return: rhs = t \oplus_id r: b x n x n
        """
        # inverts the order of the addition
        return self.addition_hrh(relations, entities)

    def get_rhs(self, triples):
        """
        :param triples: b x 3: (head, relation, tail)
        :return: b x n
        """
        tg_tails = skew_partial(self.entities[triples[:, 2]])                  # b x n x n
        return self.map_tail(tg_tails)

    def similarity_score(self, lhs, rhs):
        dist, _ = self.manifold.dist(lhs, rhs)
        return -1 * dist ** 2, dist

    def similarity_score_eval(self, lhs, rhs):
        dist, _ = self.manifold.dist_eval(lhs, rhs)
        return -1 * dist ** 2, dist

    def get_factors(self, triples):
        """
        Returns factors for embeddings' regularization.
        :param triples: b x 3: (head, relation, tail)
        :return: list of 3 tensors of b x *
        """
        heads = self.entities[triples[:, 0]]
        rel = self.relations[triples[:, 1]]        

        # extended version
        rot_params_nn = self.rot_params_nn[triples[:, 1]]

        tails = self.entities[triples[:, 2]]        

        # extended version
        return heads, rel, rot_params_nn, tails

    def compute_norms(self, points):
        entities = self.manifold.expmap_id(lalg.sym(points.detach()))
        return entities.flatten(start_dim=1).norm(dim=-1)

    def entity_norms(self):
        return self.compute_norms(self.entities)

    def relation_norms(self):
        return self.compute_norms(self.relations)

    def relation_transform_norms(self):
        return self.rel_transforms.detach().flatten(start_dim=1).norm(dim=-1)
