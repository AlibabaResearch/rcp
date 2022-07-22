from losses.supervised_losses import SupervisedL1Loss, SupervisedL2Loss, SupervisedL1RegLoss, SupervisedL1RegLossV1
from losses.unsupervised_losses import UnSupervisedL1Loss
from losses.common_losses import get_loss_weights

losses_dict = {
                'sv_l1': SupervisedL1Loss,
                'sv_l2': SupervisedL2Loss,
                'sv_l1_reg': SupervisedL1RegLoss,
                'sv_l1_reg_v1': SupervisedL1RegLossV1,
                'unsup_l1': UnSupervisedL1Loss,
               }