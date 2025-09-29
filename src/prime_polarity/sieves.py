def prime_sieve_up_to(n: int):
    """Return is_prime list (0..n) using sieve of Eratosthenes."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start : n + 1 : step] = [False] * (((n - start) // step) + 1)
        p += 1
    return is_prime


def mobius_sieve_up_to(n: int):
    """Compute Möbius mu(k) for k<=n using a sieve approach."""
    mu = [1] * (n + 1)
    is_prime = [True] * (n + 1)
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        j = 0
        while j < len(primes) and i * primes[j] <= n:
            p = primes[j]
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
            j += 1
    mu[0] = 0
    return mu


def divisors_up_to(n: int):
    """Return dict mapping m -> list of positive divisors of m for m<=n."""
    divs = {i: [] for i in range(n + 1)}
    for d in range(1, n + 1):
        for k in range(d, n + 1, d):
            divs[k].append(d)
    return divs
