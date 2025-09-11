`gaussian_process.ipuynb` Chapter 2.1, Bayesian Neural Newtork (BNN) approximates Gaussian Process (GP) under certain conditions.


Notation: $\mathbf{x}=(x_1,x_2,...,x_i,...)$

The Bayesian Neural Network is defined by
$$
\begin{cases}
f_k(\mathbf{x})&=b_k+\displaystyle\sum_{j=1}^{H}v_{jk}\cdot h_j(\mathbf{x})\\
h_j(\mathbf{x})&=\text{tanh}\left(a_j+\displaystyle\sum_{i=1}^Iu_{ij}x_i\right)
\end{cases}, \text{where}
\begin{cases}
u_{ij}&\sim\mathcal{N}(0, \sigma_u^2)\\
a_j&\sim\mathcal{N}(0, \sigma_a^2)\\
v_{jk}&\sim\mathcal{N}(0, \sigma_v^2)\\
b_k&\sim\mathcal{N}(0, \sigma_b^2)
\end{cases}
$$
We show that BNN approaches GP as $H\rightarrow\infty$ by showing

1. $h_j(x^{(p)})$ are i.i.d. for all $j$ for a fixed $x^{(p)}$
2. $v_{jk}\cdot h_j(x^{(p)})$ are i.i.d. for all $j$ for a fixed $x^{(p)}$.
3. $\text{Var}\left[v_{jk}\cdot h_j(x^{(p)})\right]<\infty$ and thus CLT applies.

Fix input to $x^{(p)}$.
<ol><li>The intuitive explanation for this is because $u_{ij}$ and $a_j$ all come from independent Gaussians, so their transformation, $h_k(x^{(p)})$, are also independent. Of course this is not obvious so I provide a rigorous proof below.

> We start by providing some theorems below.
>> **_THEOREM 1.1:_**
Let $$X:\Omega\rightarrow\mathbb{R}^d\quad\text{and}\quad Y:\Omega\rightarrow\mathbb{R}^d$$  be independent random vectors, and let $$f:\mathbb{R^d}\rightarrow\mathbb{R}\quad\text{and}\quad g:\mathbb{R}^d\rightarrow\mathbb{R}$$ be measurable functions.<br>Then $f(X)$ and $g(Y)$ are independent.
>
> <sub><sub><i>*The proof of this theorem is omitted, but rather simple and follows proof in case $d=1$.</i></sub></sub><br>
>> **_THEOREM 1.3:_** A step function is measurable.
>
>> **_THEOREM 1.2:_** A continuous function is measurable.
> 
>Since $u_{ij}$ are independent, $u_{ij}\cdot x_i^{(p)}$ are independent. Therefore, their sums, $\displaystyle\sum_{i=1}^Iu_{ij}\cdot x^{(p)}_i$, are independent (take $f_j=\sum\quad\forall j$ in Theorem 1.1). It then follows that applying $\text{tanh}$ or $\text{step}$ function preserves independence (because they are both measurable functions), i.e., $h_j(x^{(p)})$ are independent.<br><sub><sub><i>*$a_j$ are ignored in this proof, but it can be incorporated by reparameterizing $a_j=u_{(i+1)j}$ and $x^{(p)}_{i+1}=1$, much like omission of bias term in neural network parameterization.</i></sub></sub><br>


> **_NOTE:_** In fact, $h_j(x^{(p)})$ are not necessarily i.i.d. if $u_{ij}$ are not i.i.d.. To see this, consider the case where $u_{\cdot 1}$ is from a Gaussian distribtion and $u_{\cdot 2}$ are from a Uniform distirbution. Then clearly $h_1(x^{(p)})$ and $h_2(x^{(p)})$ are not i.i.d..

</li>
<li>This follows from the fact that $h_j(x^{(p)})$ are i.i.d. and $v_{jk}$ are i.i.d..<br>First of all, it is clear that all $h_j(x^{(p)})$ and $v_{jk}$ are <i>mutually</i> independent, that is, $\{h_j(x^{(p)})\}_j\cup\{v_{jk}\}_{j}$ are all independent on each other.<br>
Now it follows that $\{(h_j(x^{(p)}), v_{jk})\}_j$ are independent from the following lemma by taking $E_j=(h_j(x^{(p)}), v_{jk})$.

>**_LEMMA:_**<br>
Let $A=\displaystyle\bigcup_{n}^\circ A_n$ and $E_1,E_2,...$ be disjoint subsets of $A$ such that $\displaystyle\bigcup_i^\circ E_i\subseteq A.$ <br>
If $A_n$ are independent, then $E_i$ are independent.

Using Theorem 1.1 by taking $f,g:\mathbb{R}^2\rightarrow\mathbb{R}$ as $f(x,y)=g(x,y)=xy$, $v_{1k}\cdot h_1(x^{(p)})$ and $v_{2k}\cdot h_2{x^{(p)}}$ are independent. Repeat this and get $v_{jk}\cdot h_j(x^{(p)})$ are all independent across $j$. It is also clear that they are identically distributed because they all follow the same transformation of random variables.
</li>
<li>

We show that $Var\left[v_{jk}h_j(x^{(p)})\right]<\infty$.

$$
\begin{align*}
\mathbb{E}\left[v_{jk}\cdot h_j(x^{(p)})\right]&=\mathbb{E}\left[v_{jk}\right]\mathbb{E}\left[{h_j(x^{(p)})}\right]\quad\text{by independence}\\
&=0\quad\cdots(1)\quad\because\mathbb{E}\left[v_{jk}\right]=0\\
\text{Var}\left[v_{jk}\cdot h_j(x^{(p)})\right]&=\mathbb{E}\left[\left(v_{jk}\cdot h_j(x^{(p)})\right)^2\right]-\mathbb{E}^2\left[v_{jk}\cdot h_j(x^{(p)})\right]\\
&=\mathbb{E}\left[v_{jk}^2\right]\mathbb{E}\left[h_j(x^{(p)})^2\right] - 0\quad\quad\text{by independence, and by (1)}\\
&=\sigma_v^2\:\mathbb{E}\left[h_j(x^{(p)})^2\right]<\infty
\end{align*}
$$
Therefore CLT applies and 
$$\lim_{H\to\infty}\displaystyle\sum_{j=1}^Hv_{jk}\cdot h_j(x^{(p)})\sim\mathcal{N}\left(0,H\sigma_v^2\:\mathbb{E}\left[h_j(x^{(p)})^2\right]\right)
$$
</li>
</ol>