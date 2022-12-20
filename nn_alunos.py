"""Colocar o nome dos elementos do grupo"""

import random
import math
from audioop import add

#valor exemplificativo
alpha = 0.2
tipo_animal = ['mammal', 'bird', 'reptile', 'fish', 'amphibian', 'insect', 'invertebrate']


conf = [14, 16, 18]
alphas = [0.1, 0.3, 0.5]
iteracoes = [100, 200, 300]

def make(nx, nz, ny):
    """Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
    a criacao das diversos listas, bem como a inicializacao das listas de pesos. 
    Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
    mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
    tal como foi discutido na teorica, as saidas destas estas unidades estao sempre a -1.
    por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
    
    
    #a rede neuronal é num dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    
    
    nn = {'nx':nx, 'nz':nz, 'ny':ny, 'x':[], 'z':[], 'y':[], 'wzx':[], 'wyz':[], 'dz':[], 'dy':[]}
    
    nn['wzx'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]
    
    return nn

def sig(input):
    """Funcao de activacao (sigmoide)"""
    return 1.0/(1.0 + math.exp(- input))


def forward(nn, input):
    """Função que recebe uma rede nn e um padrao de entrada in (uma lista) 
    e faz a propagacao da informacao para a frente ate as saidas"""
    
    #copia a informacao do vector de entrada in para a listavector de inputs da rede nn  
    nn['x']=input.copy()
    nn['x'].append(-1)
    
    #calcula a activacao da unidades escondidas
    nn['z']=[sig(sum([x*w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
    nn['z'].append(-1)
    
    #calcula a activacao da unidades de saida
    nn['y']=[sig(sum([z*w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]
 
   
def error(nn, output):
    """Funcao que recebe uma rede nn com as activacoes calculadas
       e a lista output de saidas pretendidas e calcula os erros
       na camada escondida e na camada de saida"""
    
    nn['dy']=[y*(1-y)*(o-y) for y,o in zip(nn['y'], output)]
    
    zerror=[sum([nn['wyz'][i][j]*nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    
    nn['dz']=[z*(1-z)*e for z, e in zip(nn['z'], zerror)]
 
 
def update(nn, alpha):
    """funcao que recebe uma rede com as activacoes e erros calculados e
    actualiza as listas de pesos"""
    
    nn['wzx'] = [[w+x*nn['dz'][i]*alpha for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [[w+z*nn['dy'][i]*alpha for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]
    

def iterate(i, nn, input, output, alpha):
    """Funcao que realiza uma iteracao de treino para um dado padrao de entrada input
    com saida desejada output"""
    
    forward(nn, input)
    error(nn, output)
    update(nn, alpha)
    print('%03i: %s -----> %s : %s' %(i, input, output, nn['y']))

    # add_line_to_file('test.txt', '%03i: %s -----> %s : %s' %(i, input, output, nn['y']) + '\n')
    


def train_and():
    """Funcao que cria uma rede 2x2x1 e treina um AND"""
    
    net = make(2, 2, 1)
    for i in range(2000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    return net
    
def train_or():
    """Funcao que cria uma rede 2x2x1 e treina um OR"""
    
    net = make(2, 2, 1)
    for i in range(1000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1]) 
    return net

def train_xor():
    """Funcao que cria uma rede 2x2x1 e treina um XOR"""
    
    net = make(2, 2, 1)
    for i in range(10000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0]) 
    return net
    


def run():
    """Funcao principal do nosso programa, cria os conjuntos de treino e teste, chama
    a funcao que cria e treina a rede e, por fim, a funcao que a treina"""

    #test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[0]), build_sets("zoo.txt")[1])

    #Configuração 1
    add_line_to_file("test.txt", "Configuração 1: \n" )
    add_line_to_file("test.txt", "  Alpha 1: \n")
    add_line_to_file("test.txt", "      Interação 100: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[0], iteracoes[0], alphas[0]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 200: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[0], iteracoes[1], alphas[0]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 300: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[0], iteracoes[2], alphas[0]), build_sets("zoo.txt")[1])

    add_line_to_file("test.txt", "Configuração 1: \n")
    add_line_to_file("test.txt", "  Alpha 2: \n")
    add_line_to_file("test.txt", "      Interação 100: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[0], iteracoes[0], alphas[1]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 200: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[0], iteracoes[1], alphas[1]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 300: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[0], iteracoes[2], alphas[1]), build_sets("zoo.txt")[1])

    add_line_to_file("test.txt", "Configuração 1: \n")
    add_line_to_file("test.txt", "  Alpha 3: \n")
    add_line_to_file("test.txt", "      Interação 100: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[0], iteracoes[0], alphas[2]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 200: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[0], iteracoes[1], alphas[2]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 300: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[0], iteracoes[2], alphas[2]), build_sets("zoo.txt")[1])

    # Configuração 2
    add_line_to_file("test.txt", "Configuração 2: \n")
    add_line_to_file("test.txt", "  Alpha 1: \n")
    add_line_to_file("test.txt", "      Interação 100: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[1], iteracoes[0], alphas[0]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 200: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[1], iteracoes[1], alphas[0]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 300: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[1], iteracoes[2], alphas[0]), build_sets("zoo.txt")[1])

    add_line_to_file("test.txt", "Configuração 2: \n")
    add_line_to_file("test.txt", "  Alpha 2: \n")
    add_line_to_file("test.txt", "      Interação 100: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[1], iteracoes[0], alphas[1]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 200: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[1], iteracoes[1], alphas[1]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 300: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[1], iteracoes[2], alphas[1]), build_sets("zoo.txt")[1])

    add_line_to_file("test.txt", "Configuração 2: \n")
    add_line_to_file("test.txt", "  Alpha 3: \n")
    add_line_to_file("test.txt", "      Interação 100: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[1], iteracoes[0], alphas[2]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 200: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[1], iteracoes[1], alphas[2]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 300: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[1], iteracoes[2], alphas[2]), build_sets("zoo.txt")[1])

    # Configuração 3
    add_line_to_file("test.txt", "Configuração 3: \n")
    add_line_to_file("test.txt", "  Alpha 1: \n")
    add_line_to_file("test.txt", "      Interação 100: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[0], alphas[0]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 200: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[1], alphas[0]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 300: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[2], alphas[0]), build_sets("zoo.txt")[1])

    add_line_to_file("test.txt", "Configuração 3: \n")
    add_line_to_file("test.txt", "  Alpha 2: \n")
    add_line_to_file("test.txt", "      Interação 100: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[0], alphas[1]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 200: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[1], alphas[1]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 300: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[2], alphas[1]), build_sets("zoo.txt")[1])

    add_line_to_file("test.txt", "Configuração 3: \n")
    add_line_to_file("test.txt", "  Alpha 3: \n")
    add_line_to_file("test.txt", "      Interação 100: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[0], alphas[2]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 200: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[1], alphas[2]), build_sets("zoo.txt")[1])
    add_line_to_file("test.txt", "      Interação 300: \n")
    test_zoo(train_zoo(build_sets("zoo.txt")[0], conf[2], iteracoes[2], alphas[2]), build_sets("zoo.txt")[1])

    


def build_sets(f):
    """Funcao que cria os conjuntos de treino e de de teste a partir dos dados
    armazenados em f (zoo.txt). A funcao le cada linha, tranforma-a numa lista
    de valores e chama a funcao translate para a colocar no formato adequado para
    o padrao de treino. Estes padroes são colocados numa lista 
    Finalmente, devolve duas listas, uma com os primeiros 67 padroes (conjunto de treino)
    e a segunda com os restantes (conjunto de teste)"""

    lista_linhas = []
    animais = []
    padroes = []

    with open(f, encoding='UTF-8') as file:
        linhas = file.readlines()
        for linha in linhas:
            lista_linhas.append(linha.strip().split('\n'))

        for linha in lista_linhas:
            animais.append(linha[0].replace('[', "").replace(']', "").split(','))

        #Converter atributos binários extraidos em str para valores inteiros
        for animal in animais:
            for idxAtributo in range(len(animal)):
                if len(animal[idxAtributo]) == 1:
                    animal[idxAtributo] = int(animal[idxAtributo])

        padroes = translate(animais)
        random.shuffle(padroes)
        training_set = padroes[0:68]
        test_set = padroes[68:len(padroes)]

        return training_set, test_set



def translate(lista):
    """Recebe cada lista de valores e transforma-a num padrao de treino.
    Cada padrao tem o formato [nome_do_animal, padrao_de_entrada, tipo_do_animal, padrao_de_saida].
    nome_do_animal e o primeiro valor da lista e tipo_de_animal o ultimo.
    padrao_de_entrada e uma lista de 0 e 1 com os valores dos atributos.
    O numero de pernas deve tambem ser convertido numa lista de 0 e 1, concatenada com os restantes
    atributos. E.g. [0 0 0 0 1 0 0 0 0 0] -> 4 pernas.
    padrao_de_saida e uma lista de 0 e 1 que representa o tipo do animal. Tem 7 posicoes e a unica
    que estiver a 1 corresponde ao tipo do animal. E.g., [0 0 1 0 0 0 0] -> reptile.
    """

    lista_animais_formatados =[]

    for animal_n_formatado in lista:
        animal_formatado = [animal_n_formatado[0]]
        lista_atributos = []
        for idxAtributo in range(1, 13):
            lista_atributos.append(animal_n_formatado[idxAtributo])

        #formatar nº pernas
        for i in range(10):
            if i == animal_n_formatado[13]:
                lista_atributos.append(1)
            else:
                lista_atributos.append(0)

        lista_atributos.append(animal_n_formatado[14])
        lista_atributos.append(animal_n_formatado[15])
        lista_atributos.append(animal_n_formatado[16])
        animal_formatado.append(lista_atributos)

        animal_formatado.append(animal_n_formatado[-1])
        # print(len(animal_formatado))

        lista_tipo_animal = []
        indice_tipo_animal = int()
        if animal_n_formatado[-1] in tipo_animal:
            indice_tipo_animal = tipo_animal.index(animal_n_formatado[-1])
        for i in range(7):
            if i == indice_tipo_animal:
                lista_tipo_animal.append(1)
            else:
                lista_tipo_animal.append(0)

        animal_formatado.append(lista_tipo_animal)
        # print(animal_formatado)
        lista_animais_formatados.append(animal_formatado)

    return lista_animais_formatados

        

def train_zoo(training_set, conf, iteracoes, alpha):
    """cria a rede e chama a funçao iterate para a treinar. Use 300 iteracoes"""
    net = make(len(training_set[0][1]), conf, len(training_set[0][-1]))
    for i in range(iteracoes):
        for animal in training_set:
            iterate(i, net, animal[1], animal[-1], alpha)

    return net

def retranslate(out):
    """recebe o padrao de saida da rede e devolve o tipo de animal corresponte.
    Devolve o tipo de animal corresponde ao indice da saida com maior valor."""

    # idx = out.index(max(out))
    # tipo = tipo_animal[idx]

    return tipo_animal[out.index(max(out))]

def test_zoo(net, test_set):
    """Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste.
    Para cada padrao do conjunto de teste chama a funcao forward e determina o tipo
    do animal que corresponde ao maior valor da lista de saida. O tipo determinado
    pela rede deve ser comparado com o tipo real, sendo contabilizado o número
    de respostas corretas. A função calcula a percentagem de respostas corretas"""
    numero_total_test_set = len(test_set)
    numero_acertos = 0
    i = 0
    for animal in test_set:
        forward(net, animal[1])
        print(net['y'])
        tipo =  retranslate(net['y'])
        print(str(i) + " TIPO: " + tipo)
        i = i+1
        if tipo == animal[2]:
            numero_acertos += 1

    taxa_acertos = round((numero_acertos/numero_total_test_set) * 100, 2)
    print("TAXA DE ACERTO: " + str(taxa_acertos))
    add_line_to_file('test.txt', "\t\tTAXA DE ACERTO: " + str(taxa_acertos) + '\n')
    pass

def add_line_to_file(file, str):
    with open(file, 'a') as file:
        file.write(str)


if __name__ == "__main__":
    #train_and()
    run()
    # build_sets("zoo.txt")
    # train_zoo(build_sets("zoo.txt")[0])
    # test_zoo(train_zoo(build_sets("zoo.txt")[0]), build_sets("zoo.txt")[1])