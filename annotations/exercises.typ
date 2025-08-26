#import "@preview/ctheorems:1.1.3": *
#import "@preview/lovelace:0.3.0": *
#show: thmrules.with(qed-symbol: $square$)

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(languages: codly-languages, stroke: 1pt + luma(100))

#import "@preview/tablex:0.0.9": tablex, rowspanx, colspanx, cellx

#set page(width: 21cm, height: 30cm, margin: 1.5cm)

#set par(
  justify: true
)

#set figure(supplement: "Figura")

#set heading(numbering: "1.1.1")

#let theorem = thmbox("theorem", "Teorema")
#let corollary = thmplain(
  "corollary",
  "Corolário",
  base: "theorem",
  titlefmt: strong
)
#let definition = thmbox("definition", "Definição", inset: (x: 1.2em, top: 1em))
#let example = thmplain("example", "Exemplo").with(numbering: none)
#let proof = thmproof("proof", "Demonstração")

#set math.equation(
  numbering: "(1)",
  supplement: none,
)
#show ref: it => {
  // provide custom reference for equations
  if it.element != none and it.element.func() == math.equation {
    // optional: wrap inside link, so whole label is linked
    link(it.target)[(#it)]
  } else {
    it
  }
}

#set text(
  font: "Atkinson Hyperlegible",
  size: 12pt,
)

#show heading: it => {
  if it.level == 1 {
    [
      #block(
        width: 100%,
        height: 1cm,
        text(
          size: 1.5em,
          weight: "bold",
          it.body
        )
      )
    ]
  } else {
    it
  }
}


#align(center + top)[
  FGV EMAp

  João Pedro Jerônimo
]

#align(horizon + center)[
  #text(17pt)[
    Reinforcement Learning
  ]
  
  #text(14pt)[
    Exercícios do Livro
  ]
]

#align(bottom + center)[
  Rio de Janeiro

  2025
]

#pagebreak()

// ============================ PÁGINAS POSTERIORES =========================
#outline(title: "Conteúdo")

#pagebreak()

#align(center + horizon)[
  = Introduction
]

#pagebreak()

*Exercício 1.1* - _Self-Play_: Suponha que, em vez de jogar contra um oponente aleatório, o algoritmo de aprendizado por reforço descrito acima jogasse contra si mesmo, com ambos os lados aprendendo. O que você acha que aconteceria nesse caso? Ele aprenderia uma política diferente para a seleção de jogadas?

*Resolução*: _Como ambos os modelos são treinados para melhorar ao seu oponente, vai chegar um ponto que os jogos sempre darão empate, pois os modelos estarão bem treinados o suficiente para que a única forma de maximizar a recompensa seja ambos empatando_


// ===========================================================
*Exercício 1.2* - _Symmetries_:
Muitas posições do jogo-da-velha parecem diferentes, mas na verdade são iguais por causa das simetrias. Como poderíamos alterar o processo de aprendizado descrito acima para tirar proveito disso? De que maneiras essa mudança melhoraria o processo de aprendizado?
Agora pense novamente: suponha que o oponente não tirasse proveito das simetrias. Nesse caso, nós deveríamos aproveitar? É verdade, então, que posições simetricamente equivalentes devem necessariamente ter o mesmo valor?

*Resolução*: _Poderíamos restringir a quantidade de jogadas possíveis de forma a analisar os estados simétricos como os mesmos. Isso agiliza bastante o processo de treinamento do modelo, tendo em vista que a quantidade de casos a ser analisados diminuem e muito. Sim, deveríamos tirar proveito tendo em vista que algumas jogadas que levam a vitória que sejam óbvias olhando normalmente podem ser mais dificeis de ver de primeira caso o tabuleiro esteja rotacionado, além de que, pelo nosso treinamento ser mais ágil, teríamos uma vantagem em questão de desenvolvimento (Desenvolveríamos mais rápido). Sim, posições simetricamente equivalentes deveriam ter o mesmo valor_


// ===========================================================
*Exercício 1.3* - _Greedy Play_:
Suponha que o jogador de aprendizado por reforço fosse ganancioso, isto é, sempre jogasse o movimento que ele avaliasse como o melhor. Ele aprenderia a jogar melhor, ou pior, do que um jogador não ganancioso? Que problemas poderiam ocorrer?

*Resolução*: _Poderia ocorrer de que ele começasse a sempre utilizar uma mesma sequência de jogadas que foi a que maximizaram o seu ganho, de forma que ele se torne previsível, ou seja, aprenda pior_


// ===========================================================
*Exercício 1.4* - _Learning from Exploration_:
Suponha que as atualizações de aprendizado ocorressem após todos os movimentos, incluindo os movimentos exploratórios. Se o parâmetro de taxa de aprendizado for reduzido adequadamente ao longo do tempo (mas não a tendência de explorar), então os valores dos estados convergiriam para um conjunto diferente de probabilidades. Quais (conceitualmente) são os dois conjuntos de probabilidades computados quando aprendemos e quando não aprendemos a partir de movimentos exploratórios? Supondo que continuemos a fazer movimentos exploratórios, qual conjunto de probabilidades seria melhor aprender? Qual resultaria em mais vitórias?

*Resolução*: _Quando não aprendendemos com os movimentos de exploração, temos um conjunto *ótimo* de probabilidades, de forma que sempre vai levar o meu modelo ao resultado esperado. Faz sentido o melhor ser o conjunto ótimo, mesmo que possa levar a alguns casos indesejados, e eu também acredito que o que mais geraria vitórias seria esse_


// ===========================================================
*Exercício 1.5* - _Other Improvements_:
Você consegue pensar em outras maneiras de melhorar o jogador de aprendizado por reforço? Consegue pensar em alguma forma melhor de resolver o problema do jogo-da-velha, conforme foi proposto?

*Resolução*: _Eu penso em penalizar as jogadas que causam perca certa na próxima jogada, de forma que ele vai perceber que aquela jogada nunca é correta de se fazer, de tal modo que ele vai convergir para um aprendizado ideal mais rápido_


// ===========================================================