## Imitation Learning com Godot RL Agents

Imagine ensinar um robô a executar uma tarefa sem precisar programar todas as regras explicitamente. Em vez disso, ele aprende a partir de demonstrações humanas, observando como realizamos a tarefa e, então, reproduzindo o comportamento de forma autônoma. Esse é o objetivo do *Imitation Learning* (ou Aprendizado por Imitação). Na **Bonus Unit 5** do [curso de Deep Reinforcement Learning da Hugging Face](https://huggingface.co/learn/deep-rl-course/unitbonus5/introduction), é demonstrado como aplicar essa técnica de forma simples e lúdica usando o *Godot RL Agents*.

Neste texto, apresentaremos as principais seções desse tutorial, além de uma análise detalhada dos algoritmos empregados (PPO, Behavioral Cloning e GAIL), com pseudocódigos e um diagrama conceitual para ilustrar o fluxo de treinamento.

## Sumário

1. [Introduction (Introdução)](#1-introdução)  
2. [The Environment (O Ambiente)](#2-o-ambiente)  
3. [Getting Started (Primeiros Passos)](#3-primeiros-passos-getting-started)  
4. [Train our robot (Treinando nosso robô)](#4-treinando-nosso-robô)  
   - [Coleta de Demonstrações](#41-coleta-de-demonstrações)  
   - [Behavioral Cloning (BC)](#behavioral-cloning-bc)  
   - [GAIL (Generative Adversarial Imitation Learning)](#gail-generative-adversarial-imitation-learning)  
   - [PPO (Proximal Policy Optimization) e Fluxo de Treinamento](#ppo-proximal-policy-optimization-e-fluxo-de-treinamento)  
5. [Customize the environment (Personalizando o ambiente)](#5-personalizando-o-ambiente)  
6. [Conclusão](#6-conclusão)

---

## 1. Introdução

O *Imitation Learning* surge como um complemento ao *Reinforcement Learning* (RL). Enquanto o RL se baseia em recompensa e punição ao longo de múltiplas iterações para ajustar o comportamento do agente, o *Imitation Learning* busca encurtar esse caminho aprendendo diretamente a partir de exemplos. Na prática, podemos fazer um agente “assistir” a um demonstrador (que pode ser um humano ou outro agente especialista), registrar suas ações e, em seguida, reproduzir esse comportamento por meio de algoritmos como *Behavioral Cloning*.

### Vantagens

- **Velocidade de aprendizagem**: Possivelmente mais rápido, pois o agente parte de exemplos diretos.  
- **Eficácia**: Reduz a necessidade de criar funções de recompensa complexas para tarefas onde o objetivo não é trivial.  
- **Maior controle**: Os humanos têm mais controle sobre o comportamento inicial do robô.

Nesta *Bonus Unit*, o tutorial mostra como usar o Godot, um *game engine* gratuito e de código aberto, para criar ambientes onde podemos treinar agentes que aprendem por imitação. O *Godot RL Agents* é um projeto que integra algoritmos de RL (e também de Imitation Learning) ao Godot, facilitando o desenvolvimento.

---

## 2. O Ambiente

O tutorial propõe um ambiente simples em Godot, onde existe um robô capaz de se movimentar (por exemplo, um personagem dentro de uma pequena arena). A ideia é que esse robô seja controlado por um jogador humano que demonstra o comportamento desejado (andar, parar, contornar obstáculos etc.). Posteriormente, os dados coletados são usados para treinar o agente em *Behavioral Cloning*.

O **Godot RL Agents** disponibiliza scripts e ferramentas que gerenciam:

- **Coleta de dados**: registra estados, ações e recompensas (caso exista), no estilo do RL clássico.  
- **Treinamento de redes neurais**: permite a aplicação de algoritmos de Aprendizado por Imitação, como *Behavioral Cloning* e GAIL.  
- **Integração simples**: scripts prontos para orquestrar o processo de gravação e treino dentro do *game engine*.

### Fluxo de trabalho resumido

1. Criamos o nível (a cena) no Godot.  
2. Configuramos o personagem (robô) com as funcionalidades de movimento.  
3. Gravamos as ações do usuário e os estados do ambiente.  
4. Treinamos o modelo de imitação.  
5. Testamos o resultado e, se necessário, ajustamos parâmetros ou refinamos as demonstrações.

---

## 3. Primeiros Passos (Getting Started)

Para acompanhar o tutorial, você precisará:

- **Instalar o Godot** (a versão atual, recomendada no tutorial, é a 3.x ou 4.x, dependendo da compatibilidade do *Godot RL Agents*).  
- **Clonar ou baixar o repositório** do [Godot RL Agents](https://github.com/BitBasilisk/godot_rl_agents) e abrir no Godot.  
- **Familiarizar-se com a estrutura de cenas** no Godot. A *scene* principal costuma conter o cenário, o agente e scripts de controle.

### Passo a passo básico

1. **Abrir o projeto**: no Godot, clique em *Import Project*, localize o diretório do Godot RL Agents e importe-o.  
2. **Executar o projeto**: veja se tudo funciona conforme o esperado.  
3. **Identificar a cena de exemplo**: na pasta do projeto, existe um exemplo pronto de ambiente para demonstração.  

A partir disso, você está pronto para gravar suas próprias demonstrações e experimentar o *Imitation Learning*.

---

## 4. Treinando nosso robô

### 4.1 Coleta de Demonstrações

Nesta etapa, você controla o robô manualmente dentro do ambiente Godot, registrando cada ação (movimento para frente, para trás, giro à esquerda/direita). É importante executar a tarefa que você deseja ensinar ao robô de forma clara e consistente, pois quanto melhor for a “qualidade” das demonstrações, mais eficiente será o aprendizado.

**Dicas para boas demonstrações**:  
- **Variedade de situações**: inclua voltas, esquivas de obstáculos e pontos de parada.  
- **Clareza**: evite movimentos aleatórios; tente ser sistemático.  
- **Número de exemplos**: quanto mais demonstrações, maior a chance de o modelo generalizar.

---

### Behavioral Cloning (BC)

O conceito central do Imitation Learning é aproveitar a experiência de um especialista — por exemplo, um humano — para auxiliar o agente a aprender de maneira mais eficiente.

Behavioral Cloning é um dos algoritmos de Imitation Learning. Ao coletarmos a interação entre o especialista e o ambiente, obtemos a trajetória τ, composta por s (estado) e a (ação). O objetivo é atualizar a rede de forma a maximizar a probabilidade dessas trajetórias.

**Contexto de uso no script**  
- O BC é opcional e executado antes de GAIL e do RL puro.  
- No código, usamos o objeto `bc.BC(...)` da biblioteca *imitation*, fornecendo:
  - `demonstrations = trajectories` (as trajetórias gravadas).  
  - `policy = learner.policy` (ou seja, a política já associada ao PPO, que será ajustada pelos dados de demonstração).

**Ideia geral do BC**  
1. Ter um conjunto de demonstrações (estados e ações realizadas pelo especialista).  
2. Treinar uma rede neural para prever a ação que o especialista faria dado um estado (basicamente *supervised learning*).  
3. Minimizar a diferença entre a ação predita pelo agente e a ação realizada pelo especialista.

#### Pseudocódigo de Behavioral Cloning

```pseudo
algorithm Behavioral Cloning:
    input: dataset D = {(s_i, a_i)} extraído das demonstrações
    initialize policy πθ (parametrizada por θ)
    for epoch in range(num_epochs):
        for minibatch in D:
            # s, a são os pares estado-ação do especialista
            prediction = πθ(s)
            loss = L(prediction, a)  # por exemplo, cross-entropy ou MSE
            θ ← θ - α ∇(loss)
    return πθ
```
### GAIL (Generative Adversarial Imitation Learning)

O Generative Adversarial Imitation Learning (GAIL) é uma arquitetura de redes neurais profundas voltada ao Imitation Learning, em que um agente (“learner”) aprende uma habilidade observando o comportamento de outro agente (“expert”). Ela se baseia na popular arquitetura de Generative Adversarial Networks (GANs), na qual a rede é dividida em dois blocos funcionais principais que participam de um jogo adversarial: o “discriminator” (ou “critic”) aprende a distinguir exemplos reais de treinamento daqueles criados pela própria rede, enquanto o módulo “generator” (ou “actor”) aprende a produzir exemplos falsos, porém convincentes, com o objetivo de enganar o discriminador. Esses exemplos falsos são justamente o resultado útil de uma GAN.

**Contexto de uso no script**  
O GAIL é inicializado pelo objeto `GAIL(...)`, que recebe:  
- As demonstrações (`demonstrations=trajectories`).  
- Um `reward_net` (rede de recompensa diferenciável).  
- Uma referência ao PPO (`gen_algo=learner`), que será ajustado de maneira adversarial.

**Ideia geral do GAIL**  
1. Há um *discriminador (D)* que tenta dizer se uma transição (s, a) vem do especialista ou do agente.  
2. O agente (gerador, aqui usando PPO) ajusta sua política para *maximizar* a probabilidade de o discriminador classificá-lo como “especialista”.  
3. Com o tempo, se o discriminador não consegue diferenciar, o agente terá aprendido um comportamento próximo ao das demonstrações.

#### Pseudocódigo de GAIL

```pseudo
algorithm GAIL:
    initialize policy πθ
    initialize discriminator Dϕ
    
    for each iteration do:
        # 1) Atualizar o discriminador D
        coletar amostras de transições de:
           - especialista: (s^E, a^E)
           - agente: (s^A, a^A) geradas por πθ
        Dϕ é treinado para maximizar:
            L_D = E[log(Dϕ(s^E, a^E))] + E[log(1 - Dϕ(s^A, a^A))]
        
        # 2) Atualizar a política πθ (agente) via PPO, usando "recompensa" gerada por:
            r(s,a) = -log(1 - Dϕ(s, a))
        
        # Aplicar passos de RL (por ex., PPO) com reward = r(s,a)
        for update_step in range(N):
            use r(s,a) para calcular a perda de PPO
            θ ← θ - α ∇(PPO_loss)
```
### PPO (Proximal Policy Optimization) e Fluxo de Treinamento

**Contexto de uso no script**  
- PPO é instanciado pela linha `learner = PPO(...)` e serve como **política base** do agente.  
- Quando fazemos GAIL, esse PPO (chamado de “gen_algo” no script) é ajustado de forma adversarial.  
- Quando executamos RL puro (`rl_timesteps > 0`), treinamos o PPO usando as recompensas do ambiente para melhorar a política.

**Ideia Geral do PPO**  
1. Colete *rollouts* (transições estado-ação-recompensa-próximo estado) usando a política atual.  
2. Calcule a vantagem (*advantage function*) para cada ação.  
3. Atualize a política de forma a maximizar a vantagem, mas mantendo a nova política próxima (via *clipping*) da política antiga.

#### Pseudocódigo de PPO

```pseudo
algorithm PPO:
    initialize policy πθ (parametrizada por θ)
    for each iteration do:
        collect a set of trajectories τ = { (s_t, a_t, r_t, s_{t+1}) } using πθ
        compute advantages A_t e retorno G_t
        
        for each epoch in range(K):
            for each minibatch do:
                # probabilidades atuais e antigas
                ratio = prob_current / prob_old
                
                # perda "clipped"
                L_clip = min( ratio * A_t , clip(ratio, 1-ε, 1+ε) * A_t )
                
                loss = - mean( L_clip ) + terms_de_regularização
                θ ← θ - α ∇(loss)
        update πθ_old ← πθ
    return πθ
```
#### Diagrama de Aplicação

A figura abaixo mostra como esses algoritmos se encaixam no fluxo de treinamento definido no *script*:

            ┌──────────────────────────┐
            │      Demonstrações       │
            │   (trajetórias expert)   │
            └────────────┬─────────────┘
                         │
              [Se bc_epochs > 0]  
                         ▼
            ┌───────────────────────────┐
            │     Behavioral Cloning    │
            │  (Treina Política Inicial)│
            └────────────┬──────────────┘
                         │
              [Se gail_timesteps > 0]
                         ▼
            ┌──────────────────────────┐
            │           GAIL           │
            │Discriminator vs. Gerador │
            └────────────┬─────────────┘
                         │
              [Se rl_timesteps > 0]
                         ▼
            ┌──────────────────────────┐
            │           PPO            │
            │  (RL puro no ambiente)   │
            └────────────┬─────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │   Política      │
               │   (agent.zip)   │
               └─────────────────┘

1. **Carregamos as demonstrações** e, se `bc_epochs > 0`, treinamos a política inicial com *Behavioral Cloning*.  
2. Se `gail_timesteps > 0`, partimos dessa política (já imitada) e executamos o GAIL, que usa um discriminador para aproximar ainda mais o comportamento especialista.  
3. Se `rl_timesteps > 0`, refinamos a política treinando o PPO diretamente no ambiente, usando as recompensas normais do próprio ambiente.  
4. O resultado final é uma **política otimizada** que, em tese, combina a vantagem de aprender com demonstrações (BC/GAIL) e a exploração/otimização adicional via *Reinforcement Learning* puro (PPO).

---

## 5. Personalizando o ambiente

Um dos grandes atrativos do Godot é a facilidade de customização. Para adaptar o ambiente aos seus objetivos, considere:

- **Alterar o cenário**: inserindo obstáculos diferentes, mudando o layout do terreno ou até acrescentando elementos interativos (por exemplo, itens colecionáveis).  
- **Mudar a dinâmica do robô**: pode-se ajustar velocidade máxima, aceleração ou até mesmo incluir novos modos de locomoção.  
- **Adicionar sensores**: o robô pode ter sensores de proximidade, visão ou áudio, ampliando a complexidade do aprendizado.

### Ajuste de parâmetros

Além das variáveis do ambiente, existem **hiperparâmetros** no treinamento (típicos de redes neurais) que podem ser ajustados:

- *Learning rate*  
- Tamanho do *batch*  
- Número de *epochs*  
- Arquitetura da rede (número de camadas, neurônios, etc.)

Essas configurações podem afetar diretamente a qualidade e a velocidade de convergência do modelo. A *trial and error* (tentativa e erro) é comum nessa etapa, pois depende bastante da natureza dos dados e do comportamento que se deseja aprender.

---

## 6. Conclusão

O *Imitation Learning* é uma forma poderosa de ensinar agentes, poupando tempo e esforço na hora de definir recompensas complexas. No contexto do Godot, a abordagem é especialmente atraente por ser mais visual e interativa. O *Godot RL Agents* oferece um ecossistema pronto para experimentação, onde é possível gravar demonstrações, treinar modelos de *Behavioral Cloning* e executar testes para avaliar o desempenho do agente.

Na prática, você:

1. Constrói ou carrega um ambiente no Godot.  
2. Coleta demonstrações controlando um robô manualmente.  
3. Usa os dados para treinar um modelo de imitação (BC e/ou GAIL).  
4. Testa o modelo e refina-o (mais dados, outras configurações, novos sensores, etc.).  
5. (Opcional) Complementa o treinamento com **RL puro** via PPO, caso deseje ir além do que foi demonstrado.

Assim, o caminho para criar um agente personalizado, capaz de aprender comportamentos específicos, torna-se muito mais simples. Esse método é especialmente útil em cenários nos quais a definição de uma função de recompensa seja complexa ou em tarefas que requerem uma “intuição” mais humana.

Para criar os exemplos do especialista, foram gerados 24 episódios completando todo o circuito, sem falhas. Nos momentos em que era necessário aguardar, fizemos o robô ficar girando e ele aprendeu também este comportamento, embora não realize na mesma velocidade realizada pelo expert. 

## Demonstração em vídeo

[![Veja a demonstração](https://img.youtube.com/vi/BYtNCuhPn44/0.jpg)](https://www.youtube.com/watch?v=BYtNCuhPn44)

Se você ficou interessado, vale a pena conferir a [Bonus Unit 5 do Curso de Deep RL da Hugging Face](https://huggingface.co/learn/deep-rl-course/unitbonus5/introduction) e se aprofundar na documentação do [Godot RL Agents](https://github.com/BitBasilisk/godot_rl_agents). Você descobrirá todo um mundo de possibilidades para criar soluções de inteligência artificial, seja em jogos, robótica ou aplicações de simulação.

