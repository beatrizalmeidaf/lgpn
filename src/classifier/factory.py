import torch
from classifier.r2d2 import R2D2
from classifier.mbc import MBC
from dataset.utils import tprint

def get_classifier(ebd_dim, args):
    '''
    Cria e retorna uma instância de um modelo classificador com base nos argumentos fornecidos.
    - Seleciona o tipo de classificador ('r2d2' ou 'mbc') com base em args.classifier.
    - Se um snapshot (modelo pré-treinado) for fornecido em args.snapshot, carrega os pesos no modelo.
    - Move o modelo para a gpu especificada em args.cuda, se houver.
    - Retorna o modelo classificador configurado.
    '''

    tprint("Contruindo Classificador")
    # import pdb
    # pdb.set_trace()

    # verifica qual classificador foi especificado nos argumentos e o instancia
    if args.classifier == 'r2d2':
        model = R2D2(ebd_dim, args)
    elif args.classifier == 'mbc':
        model = MBC(ebd_dim, args)
    else:
        # lança um erro se o nome do classificador for inválido
        raise ValueError('Classificador inválido'
                         'classificador só pode ser: mbc, r2d2.')

    # verifica se há um snapshot de modelo pré-treinado para carregar
    if args.snapshot != '':
        # carregar modelos pré-treinados
        tprint("Carregando classificador pré treinado de {}".format(
            args.snapshot + '.clf'
        ))
        # carrega o dicionário de estado (pesos) do arquivo .clf
        model.load_state_dict(torch.load(args.snapshot + '.clf'))

    # se uma gpu foi especificada, move o modelo para a gpu
    if args.cuda != -1:
        return model.cuda(args.cuda)
    # caso contrário, retorna o modelo na cpu
    else:
        return model