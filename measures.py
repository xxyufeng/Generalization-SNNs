import torch
import math
import scipy.sparse
import numpy as np
from scipy.sparse.linalg import svds

# This function calculates various measures on the given model and returns a dictionary whose keys are the measure names
# and values are the corresponding measures on the model
def calculate(model, init_model, device, train_loader, margin):

    # switch to evaluate mode
    model.eval()
    init_model.eval()

    modules = list(model.children())
    init_modules = list(init_model.children())

    D = modules[0].weight.size(1) # data dimension
    m = modules[0].weight.size(0) # number of hidden units
    C = modules[2].weight.size(0) # number of classes (output dimension)
    num_param = sum(p.numel() for p in model.parameters()) # number of parameters of the model
    beta = 1 / (m ** 0.25) #controlling factor of Theorem 3

    margin = 1
    G = math.sqrt(2) / margin if C > 1 else 1 #/ margin # Lipschitz constant of loss function
    G_act = 1 # Lipschitz constant of activation function
    b = 1 # maximum of the loss function

    with torch.no_grad():
        # Eigenvalues of the weight matrix in the first layer
        _,S1,_ = modules[0].weight.svd()
        # Eigenvalues of the weight matrix in the second layer
        _,S2,_ = modules[2].weight.svd()
        # Eigenvalues of the initial weight matrix in the first layer
        _,S01,_ = init_modules[0].weight.svd()
        # Eigenvalues of the initial weight matrix in the second layer
        _,S02,_ = init_modules[2].weight.svd()
        # Frobenius norm of the weight matrix in the first layer
        Fro1 = modules[0].weight.norm()
        # Frobenius norm of the weight matrix in the second layer
        Fro2 = modules[2].weight.norm()
        # difference of final weights to the initial weights in the first layer
        diff1 = modules[0].weight - init_modules[0].weight
        # difference of final weights to the initial weights in the second layer
        diff2 = modules[2].weight - init_modules[2].weight
        # Euclidean distance of the weight matrix in the first layer to the initial weight matrix
        Dist1 = diff1.norm()
        # Euclidean distance of the weight matrix in the second layer to the initial weight matrix
        Dist2 = diff2.norm()
        # L_{1,infty} norm of the weight matrix in the first layer
        L1max1 = modules[0].weight.norm(p=1, dim=1).max()
        # L_{1,infty} norm of the weight matrix in the second layer
        L1max2 = modules[2].weight.norm(p=1, dim=1).max()
        # L_{2,1} distance of the weight matrix in the first layer to the initial weight matrix
        L1Dist1 = diff1.norm(p=2, dim=1 ).sum()
        # L_{2,1} distance of the weight matrix in the second layer to the initial weight matrix
        L1Dist2 = diff2.norm(p=2, dim=1 ).sum()

        # L_{1,2} distance of the weight matrix in the first layer to the initial weight matrix
        L2Dist1 = diff1.norm(p=2, dim=0 ).norm(p=1)
        # L_{1,2} distance of the weight matrix in the second layer to the initial weight matrix
        L2Dist2 = diff2.norm(p=2, dim=0 ).norm(p=1)

        # Calculation of path norm and standard path norm
        L2_rows1 = diff1.norm(p=2, dim=1)
        L2_rows1_1 = modules[0].weight.norm(p=2, dim=1)
        col_sum2 = modules[2].weight.norm(p=1, dim=0)
        product_path_norm = L2_rows1 * col_sum2
        product_standard_path_norm = L2_rows1_1 * col_sum2



        measure = {}
        measure['Frobenius1'] = Fro1
        measure['Frobenius2'] = Fro2
        measure['Distance1'] = Dist1
        measure['Distance2'] = Dist2
        measure['Spectral1'] = S1[0]
        measure['Spectral2'] = S2[0]
        measure['L12_Spec'] = L2Dist1 * S2[0]
        measure['Fro_Fro_c'] = Dist1 * Fro2 * math.sqrt(C)
        measure['Dist_Spec'] = S2[0] * Dist1 * math.sqrt(m)
        measure['Dist_Fro'] = Dist1 * Fro2
        measure['standard_path_norm'] = product_standard_path_norm.sum()
        measure['path_norm'] = product_path_norm.sum()

        # delta is the probability that the generalization bound does not hold
        delta = 0.01
        # m is the number of training samples
        n = len(train_loader.dataset)
        measure['#training_samples'] = n
        layer_norm, data_L2, data_Linf, domain_L2 = 0, 0, 0, 0
        data_r1mat = 0
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device).view(target.size(0),-1)
            layer_out = torch.zeros(target.size(0), m).to(device)

            # calculate the norm of the output of the first layer in the initial model
            def fun(m, i, o): layer_out.copy_(o.data)
            h = init_modules[1].register_forward_hook(fun)
            output = init_model(data)
            layer_norm += layer_out.norm(p=2, dim=0) ** 2
            h.remove()

            # L2 norm squared of the data
            data_L2 += data.norm() ** 2
            # maximum L2 norm squared on the training set. We use this as an approximation of this quantity over the domain
            domain_L2 = max(domain_L2, data.norm(p=2, dim = 1).max() ** 2)
            # L_infty norm squared of the data
            data_Linf += data.max(dim = 1)[0].max() ** 2
            # Rank 1 matrix of X: XX^T
            data_r1mat += data.t().mm(data)

        # computing the average
        data_L2 /= n
        data_Linf /= n
        layer_norm /= n

        # computing R1 and R1_shamir
        measure['R_1'] = Fro2 * math.sqrt(layer_norm.sum()) / math.sqrt(n)
        measure['R_1_shamir'] = G_act * Fro2 * math.sqrt(domain_L2) * S01[0] / math.sqrt(n)
        measure['measure_R1'] = measure['R_1'] * n * math.sqrt(C) / math.sqrt(data_L2 * n)

        # spectral norm of the rank 1 matrix
        try:
            eigvals = torch.linalg.eigvalsh(data_r1mat.cpu())
            data_r1mat_spec = float(eigvals.max().item())
        except Exception:
            # fallback to SVD largest singular value
            data_r1mat_spec = float(torch.linalg.svdvals(data_r1mat.cpu())[0].item())
        measure['data_L2'] = data_L2 * n
        measure['data_r1mat'] = data_r1mat_spec

        # number of parameters
        measure['#parameter'] = num_param

        # Generalization bound based on the VC dimension by Harvey et al. 2017
        VC = (2 + num_param * math.log(8 * math.e * ( m + 2 * C ) * math.log( 4 * math.e * ( m + 2 * C ) ,2), 2)
                * (2 * (D + 1) * m + (m + 1) * C) / ((D + 1) * m + (m + 1) * C))
        measure['VC capacity'] = 8 * (C * VC * math.log(math.e * max(n / VC, 1))) + 8 * math.log(2 / delta)
        measure['VC generalization'] = math.sqrt((8 * (C * VC * math.log(math.e * max(n / VC, 1))) + 8 * math.log(2 / delta)) / n)

        # Generalization bound by Bartlett and Mendelson 2002
        R = 8 * C * L1max1 * L1max2 * 2 * math.sqrt(math.log(D)) * math.sqrt(data_Linf) / margin
        measure['L1max capacity'] = (R + 3 * math.sqrt(math.log(n / delta))) ** 2
        measure['L1max generalization'] = (R + 3 * math.sqrt(math.log(n / delta))) / math.sqrt(n)

        # Generalization bound by Golowich et al. 2018 (for 2-layer NN and binary classification)
        R = 2 * math.sqrt(2) * (2 * math.sqrt(math.log(2)) + 1) * math.sqrt(C) * Fro1 * Fro2 * math.sqrt(data_L2) / margin
        measure['Fro capacity'] = (R + 3 * math.sqrt(math.log(n / delta))) ** 2
        measure['Fro generalization'] = (R + 3 * math.sqrt(math.log(n / delta))) / math.sqrt(n)

        # Generalization bound by Bartlett et al. 2017
        R = (144 * math.log(n) * math.log(2 * num_param) * (math.sqrt(data_L2) + 1 / math.sqrt(n))
                * (((S2[0] * L1Dist1) ** (2 / 3) + (S1[0] * L1Dist2) ** (2 / 3) ) ** (3 / 2)) / margin)
        measure['Spec_L1 capacity'] = (R + math.sqrt(4.5 * math.log(1 / delta) + math.log(2 * n / max(margin, 1e-16))
                                    + 2 * math.log(2 + math.sqrt(n * data_L2)) + 2 * math.log( (2 + 2 * Dist1)
                                        * (2 + 2 * Dist2) * (2 + 2 * S1[0]) * (2 + 2 * S2[0])))) ** 2
        measure['Spec_L1 generalization'] = (R + math.sqrt(4.5 * math.log(1 / delta) + math.log(2 * n / max(margin, 1e-16))
                                    + 2 * math.log(2 + math.sqrt(n * data_L2)) + 2 * math.log( (2 + 2 * Dist1)
                                        * (2 + 2 * Dist2) * (2 + 2 * S1[0]) * (2 + 2 * S2[0])))) / math.sqrt(n)

        # Generalization bound by Neyshabur et al. 2018
        R = (42 * 8 * S1[0] * math.sqrt(math.log(8 * m)) * math.sqrt(domain_L2)
            * math.sqrt(m * (S2[0] * Dist1) ** 2 + C * (S1[0] * Dist2) ** 2 ) / (math.sqrt(2) * margin))
        measure['Spec_Fro capacity'] = R ** 2 + 6 * math.log( 2 * n / delta )
        measure['Spec_Fro generalization'] = math.sqrt(R ** 2 + 6 * math.log( 2 * n / delta ) ) / math.sqrt(n)

        # Generalization bound by Neyshabur et al. 2019
        R = (3 * math.sqrt(2) * (math.sqrt(2 * C) + 1) * (Fro2 + 1)
            * (math.sqrt(layer_norm.sum()) + (Dist1 *  math.sqrt(data_L2)) + 1 ) / margin)
        measure['Neyshabur capacity'] = (R + 3 * math.sqrt((5 * m + math.log(max(1, margin * math.sqrt(n)) / delta)))) ** 2
        measure['Neyshabur generalization'] = (R + 3 * math.sqrt((5 * m + math.log(max(1, margin * math.sqrt(n)) / delta)))) / math.sqrt(n)

        # Generalization bound by Magen et al. 2023
        R = 2 * (4 + (2*2*math.sqrt(2)) * G_act * (Fro2+1) * (Dist1+1) * math.sqrt(domain_L2) * math.log(n) * (1 + S01[0] * math.sqrt(domain_L2) + math.log((Dist1+1) * math.sqrt(domain_L2))) + (G_act * S01[0] * (Fro2+1) * math.sqrt(domain_L2)))
        R_2 = 3 * b * math.sqrt(math.log(2 * (Dist1 + 1) * (Fro2 + 1) * (Dist1 + 2) * (Fro2 + 2) / delta)) / math.sqrt(2)
        measure['Magen capacity'] = (R + R_2) ** 2
        measure['Magen generalization'] = (R + R_2) / math.sqrt(n)

        # Our generalization bound using standard path norm (for 2-layer NN and binary classification)
        R = 4 * math.sqrt(C) * (measure['standard_path_norm'] + 1) * math.sqrt(data_L2)
        R_2 = 3 * b * math.sqrt(math.log(2 * (Dist1 + 1) * (Fro2 + 1) * (Dist1 + 2) * (Fro2 + 2) * (measure['standard_path_norm']+1) * (measure['standard_path_norm']+2)/ delta)) / math.sqrt(2)
        measure['Our capacity (std path norm)'] = (R + R_2) ** 2
        measure['Our generalization (std path norm)'] = (R + R_2) / math.sqrt(n)

        # Our generalization bound 
        c_r1r2 = 2 * math.sqrt(2) * (1 + 1/ (2 * math.log(2 * m * C))) * math.sqrt(math.log(2 * m * C * math.ceil(math.log(max(math.sqrt(5 * m), math.sqrt(5 * m * C) * (Dist1+1) * (Fro2 + 1) / (2*beta)),2))))
        # an factor \sqrt{2} can be ignored in R if we consider the binary classification case
        # R_1 = 2 * G * (Fro2 + 1) * math.sqrt(C * layer_norm.sum())
        # R = 2 * G * G_act * max(2*beta, 2*measure['path_norm']) * (3 * math.sqrt(data_L2) + c_r1r2 * math.sqrt(data_r1mat_spec / n))
        # R_2 = 3 * b * math.sqrt(math.log(2 * (Dist1 + 1) * (Fro2 + 1) * (Dist1 + 2) * (Fro2 + 2) * max(2, 2*measure['path_norm']/beta) / delta)) / math.sqrt(2)
        
        # multi-class classification case
        R_1 = 2 * math.sqrt(2) * G * (Fro2 + 1) * math.sqrt(C * layer_norm.sum())
        R = 2 * math.sqrt(2) * G * G_act * max(2*beta, 2*measure['path_norm']) * (3 * math.sqrt(data_L2) + c_r1r2 * math.sqrt(data_r1mat_spec / n))
        R_2 = 3 * b * math.sqrt(math.log(2 * (Dist1 + 1) * (Fro2 + 1) * (Dist1 + 2) * (Fro2 + 2) * max(2, 2*measure['path_norm']/beta) / delta)) / math.sqrt(2)
        measure['Our capacity'] = (R_1+ R + R_2) ** 2
        measure['Our generalization'] = (R_1 + R + R_2) / math.sqrt(n)

        ##simplified bounds
        # (1) Bartlett et al. 2019
        measure['Bartlett_2019 simp'] = math.sqrt(D * m)
        # (2) Bartlett et al. 2002
        measure['Bartlett_2002 simp'] = modules[0].weight.abs().max(dim=0)[0].sum() * modules[2].weight.abs().max(dim=0)[0].sum()
        # (3) Neyshabur et al. 2015
        measure['Neyshabur_2015 simp'] = measure['standard_path_norm']
        # (4) Golowich et al. 2018
        measure['Golowich_2018 simp'] = (Fro1 * Fro2)
        # (5) Bartlett et al. 2017
        measure['Bartlett_2017 simp'] = (S1[0] * L2Dist2) + (S2[0] * L2Dist1)
        # (6) Neyshabur et al. 2018
        measure['Neyshabur_2018 simp'] = S1[0] * Dist2 + math.sqrt(m) * S2[0] * Dist1
        # (7) Neyshabur et al. 2019
        measure['Neyshabur_2019 simp'] = S01[0] * Fro2 + Dist1 * Fro2 + math.sqrt(m)
        # (8) Magen et al. 2023
        measure['Magen_2023 simp'] = 1 / math.sqrt(domain_L2) + Fro2 * (S01[0] + Dist1 * (1 + S01[0] * math.sqrt(domain_L2))) #Dist1 * Fro2 * (S01[0] + 1)
        # (9) Daniely et al. 2024
        measure['Daniely_2024 simp'] = S01[0] * Fro2 + Dist1 * Fro2
        # Ours
        measure['Our simp'] = measure['path_norm']



    return measure

def cal_path_norm_distribution(model, init_model):
    """
    Calculate the distribution of ||w_j - w^0_j|| / |v_j|
    """
    model.eval()
    init_model.eval()
    with torch.no_grad():
        modules = list(model.children())
        init_modules = list(init_model.children())
        
        diff1 = modules[0].weight - init_modules[0].weight
        L2_rows1 = diff1.norm(p=2, dim=1)
        col_sum2 = modules[2].weight.norm(p=1, dim=0)
        
        ratio = L2_rows1 / (col_sum2 + 1e-12)
        return ratio.cpu().numpy()

