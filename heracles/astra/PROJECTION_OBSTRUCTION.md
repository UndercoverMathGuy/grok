# Junk can be projected out of predictions, but not out of their representation cost

Status: analytic proof. No numerical searches or probabilistic sampling are used.

The clean projection preserves or improves the raw margin. Nevertheless, for every
fixed nu>q>=2 and every sufficiently large odd prime p, there is an explicit positive-
margin network whose normalized margin is STRICTLY BETTER than that of every network
representing its clean projection. Each neuron in the construction is single-frequency;
different neurons use different frequencies. Mixed-frequency neurons are not needed.

This refutes universal spectrum-preserving purification, including for feasible networks.
It does not refute the ansatz's unrestricted global optimality: section 6 explicitly
constructs a clean network with a DIFFERENT spectrum that beats this family when p is
sufficiently large. This gives an analytic version of the dense-junk behavior discussed
in opus55/README.md, with explicit phases and without relying on its numerical evidence.

## 1. The prediction-level projection works exactly

Let n=nu-1, y(x)=sum_j x_j in Z_p, and define

    (Pi Z)(x,c) = p^(-n) sum_{t in Z_p^n} Z(x+t,c+sum_j t_j).

Every term in the projected gap is an original valid gap:

    (Pi Z)(x,y)-(Pi Z)(x,c)
      = average_t [Z(x+t,y+sum t)-Z(x+t,c+sum t)] >= m(Z).

Therefore m(Pi Z)>=m(Z). On a Fourier component (a_1,...,a_n,b), averaging retains
the component exactly when a_j=-b for every j. Thus Pi removes all nonclean tensor
components; the b=0 constant is irrelevant to gaps and can be centered away.

The missing implication is about R_q(Z), the minimum P_q over all representations of Z.
Neither R_q(Pi Z)<=R_q(Z) nor monotonicity of m/R_q^(nu/q) under Pi follows from this
margin inequality. Sections 2-5 show both are false in general, even when m(Z)>0.

## 2. An explicit family with dense clean spectrum and controlled junk

Fix nu>q>=2, n=nu-1, r=q/nu. Let p>max(n,3) be an odd prime and h=(p-1)/2.
Use one neuron for each k=1,...,h. Set

    alpha_k = pi*k^3/p,
    E_kj(x) = cos(2*pi*k*x/p + alpha_k),       j=1,...,n,
    W_k(c)  = cos(2*pi*k*c/p + n*alpha_k).

Call its logit tensor Z. The squared norm of each neuron is nu*p/2, so its actual
penalty is

    P_0 = h*(nu*p/2)^(q/2).                                   (2.1)

With normalized Fourier coefficients, the clean coefficient at every positive frequency
is G_k=2^(-nu), independent of alpha_k. Thus

    (Pi Z)(x,c)=2^(1-nu) sum_{k=1}^h cos(2*pi*k*(c-y)/p).

For every nonzero offset d, sum_k cos(2*pi*k*d/p)=-1/2. Every projected gap is therefore

    m_0 = p/2^nu.                                            (2.2)

The junk coefficient associated with a nonempty subset T at frequency k is

    J_{k,T}=2^(-nu) exp(2*pi*i*|T|*k^3/p).

These are nonzero, full-sized junk coefficients. Their cubic phases will keep their
combined effect on any individual gap much smaller than m_0 when p is large.

## 3. An elementary bound for the cubic sums

Write e_p(t)=exp(2*pi*i*t/p), and define

    S(a,b)=sum_{k in Z_p} e_p(a*k^3+b*k),    a!=0.

For p>3,

    |S(a,b)| <= L_p := sqrt(p+(p-1)*sqrt(p)) <= sqrt(2)*p^(3/4). (3.1)

Here is a complete elementary proof. First, for A!=0, a quadratic sum
Q(A,B)=sum_k e_p(A*k^2+B*k) has |Q(A,B)|^2=p: expanding the square with difference
variable t leaves an inner sum sum_k e_p(2A*t*k), which vanishes unless t=0.

For the cubic sum, the same difference expansion gives

    |S(a,b)|^2
      = sum_{t in Z_p} e_p(a*t^3+b*t)
          sum_{k in Z_p} e_p(3a*t*k^2+3a*t^2*k).

The t=0 term is p. Every other inner sum has magnitude sqrt(p), since 3a*t!=0.
The triangle inequality proves (3.1). No deep exponential-sum estimate is assumed.

## 4. Positive margin, uniformly over the full table

For a row x and offset d!=0, expand all cosine products. The clean gap is m_0.
The junk associated with T contributes

    2^(-nu) [S(|T|,2 sum_{j in T} x_j)
             -S(|T|,2 sum_{j in T} x_j+d)].

To see why this is the COMPLETE sum, the positive-frequency term has phase
|T|*k^3+2k*sum_T x_j. Its conjugate is the negative-frequency term, because this
polynomial is odd in k. The k=0 terms cancel between the two sums.

The hypothesis p>n ensures |T|!=0 mod p. Applying (3.1) to all 2^n-1 nonempty
subsets proves

    |gap_Z(x,d)-m_0| <= 2^(1-nu) (2^n-1) L_p,
    m(Z) >= (1-epsilon_p) m_0,
    epsilon_p = 2(2^n-1)L_p/p <= 2sqrt(2)(2^n-1)p^(-1/4).     (4.1)

For fixed nu, epsilon_p tends to zero. In particular the explicit network has positive
margin at every sufficiently large prime, without any search over phases or weights.

## 5. Cleaning this network necessarily costs more

The exact junk-free feature-cost theorem and neuron budget from ELEGANT.md imply
that EVERY representation of Pi Z obeys

    P_q >= (2p*nu)^(q/2) nu^(1-r) sum_{k=1}^h |G_k|^r
         = nu^(1-r) h*(nu*p/2)^(q/2)
         = nu^(1-r) P_0.                                    (5.1)

The bound is attained: replace each original neuron by a nu-neuron common-phase
block, with each slot scaled by nu^(-1/nu), to retain the same clean coefficient.
Therefore

    R_q(Pi Z)=nu^(1-r)P_0 > P_0 >= R_q(Z),                   (5.2)

where R_q means minimum representation penalty. In particular, this is not just an
inefficient implementation of the projection; its minimum possible cost increases.

Moreover, comparing the actual constructed network with the BEST implementation of its
clean projection gives

    [m(Z)/P_0^(1/r)] / [m_0/R_q(Pi Z)^(1/r)]
       >= (1-epsilon_p) nu^((1-r)/r).                       (5.3)

Since q<nu, choose p large enough that

    epsilon_p < 1-nu^(-(nu-q)/q).

Then the right side exceeds one. Deleting the junk improves raw margin but strictly
WORSENS normalized margin, even after optimally re-representing the clean result.

For a concrete parameter range, take nu=3,q=2 and any prime p>=2^20. Then
epsilon_p<=3sqrt(2)/16<1/3. The original network has normalized margin more than
(2/3)sqrt(3)=2/sqrt(3)>1 times the best value of its clean projection.

This establishes the obstruction by explicit weights and analytic inequalities. The
large lower bound on p is a sufficient bound from the elementary estimate, not a claim
that the phenomenon first occurs there.

## 6. A different clean spectrum beats this family at large p

This point can also be made constructively, without any random choice. Let
K=ceil(log_2 p), and use the frequency list 1,2,4,...,2^(K-1), reduced modulo p.
For every nonzero d, start with its least absolute residue d_0 in [1,p/2]. Repeatedly
double it until it first reaches [p/4,p/2]. This takes fewer than K steps and involves
no wraparound before reaching that interval. Hence for at least one frequency in
the list, 1-cos(2*pi*k*d/p)>=1.

At each listed frequency build a clean nu-neuron block with coefficient 2^(-nu).
Duplicates in the list do not hurt the construction. Its margin is at least 2^(1-nu),
and its penalty is at most K*nu^(1-r)*(nu*p/2)^(q/2).

The original dense-junk network has margin at most m_0, since its row-average gap
at every offset is m_0. Therefore the ratio of this sparse clean network's normalized
margin to the dense-junk network's normalized margin is at least

    (2/p) * [h/(K*nu^(1-r))]^(1/r).

This is greater than one whenever

    2^r h > p^r K nu^(1-r).                                  (6.1)

For fixed nu and q<nu, the ratio of the left side to the right side tends to infinity:
it grows proportionally to p^(1-r)/log p. Thus all sufficiently large primes satisfy
both (6.1) and section 5's strict projection obstruction.

So, in this explicit family, changing the clean spectrum is essential: deleting its
junk at fixed clean spectrum loses, while a different clean spectrum wins.

## 7. What this settles about a simplification or contradiction

The output projection is an exact simplification: after it, all junk is gone and raw
margin has not decreased. The obstruction is entirely in representation cost.

The following proposals cannot be universal lemmas:

* Delete junk while preserving the clean spectrum and not increasing P_q.
* Project to clean logits and compensate for any extra cost using the margin improvement.
* Derive either conclusion merely from positive margin or Fourier-pure neurons.

An optimizer-specific contradiction is not ruled out. A global optimum might have a
spectral structure that excludes the construction above. Alternatively, a successful
replacement could change the clean spectrum completely. Neither possibility is proved
here. Replacing the network by a different clean spectrum is the essential freedom that
must remain in a general global-optimality argument.
