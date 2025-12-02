# def vit_param_groups(vit_model, base_lr=5e-5, head_lr=1e-3, weight_decay=0.05):
#     head_params, backbone_params = [], []
#     for n, p in vit_model.named_parameters():
#         if not p.requires_grad:
#             continue
#         if n.startswith('head') or '.head.' in n:
#             head_params.append(p)
#         else:
#             backbone_params.append(p)
#     return [
#         {'params': backbone_params, 'lr': base_lr, 'weight_decay': weight_decay},
#         {'params': head_params,     'lr': head_lr,  'weight_decay': weight_decay},
#     ]
def vit_param_groups(vit_model, base_lr=5e-5, head_lr=1e-3, head1_lr=1e-3, weight_decay=0.05):
    head_params, head1_params, backbone_params = [], [], []
    for n, p in vit_model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('head1') or '.head1.' in n:
            head1_params.append(p)
        elif n.startswith('head') or '.head.' in n:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {'params': backbone_params, 'lr': base_lr,  'weight_decay': weight_decay},
        {'params': head_params,     'lr': head_lr,  'weight_decay': weight_decay},
        {'params': head1_params,    'lr': head1_lr, 'weight_decay': weight_decay},
    ]
