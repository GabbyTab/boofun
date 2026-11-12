import re
import pickle
import operator as op
import math
import string
import urllib
import sys
from numpy import sign
from math import *
import random
import time
#del sum
#del pow
#del all
## import PIL.Image as IMG

blockify = lambda l,n: [l[i:i+n] for i in range(0,len(l),n)]

unique = lambda l: list(set(l))


def sum_list(l):
    ret = []
    for x in l:
        ret.extend(x)
    return ret
#sum_list = lambda l: sum(l,[])

tuplize = lambda l: tuple(sorted(l))

def popcnt(x):
    return bin(x)[2:].count("1")

def poppar(x):
    return popcnt(x)%2

def over(n,k):
#    if k>n:
#        return 0
#    if n<0 or k<0:
#        return 0
    res = 1
    for i in range(1,k+1):
        res = (res * (n-i+1)) // i
    return res

def over_le(n,k):
    if k>n:
        return 0
    if n<0 or k<0:
        return 0
    x = 1
    ret = 1
    for i in range(1,k+1):
        x = (x * (n-i+1)) // i
        ret += x
    return ret

def binset(n):
    x = bin(n)[2:][::-1]
    return set([i for i in range(len(x)) if x[i]=='1'])

def num2bin_list(num, n_digits):
    return [(num>>i)&1 for i in range(n_digits-1, -1, -1)]

def factorial(n):
    return reduce(lambda x,y:x*y,range(1,n+1)+[1])

def out(l):    
    if isinstance(l,dict):
        l = sorted(l.items())
        for x in l:
            print("%s : %s" % (x[0],x[1]))
        return
    if len(l)>10:
        formt = "%" + str(int(math.ceil(math.log(len(l), 10))))+ "d: %s"
    else:
        formt = "%d: %s"
    for i,x in enumerate(l):
        print(formt % (i,x))

def bits(i,n):
    return map(int,bin((1<<n) | (i&((1<<n)-1)))[3:])

def save(obj,filename):
    pickle.dump(obj,file(filename,"wb"))

def load(filename):
    return pickle.load(file(filename,"rb"))

def count(lst):
    d = {}
    for x in lst:
        d[x] = d.get(x,0) + 1
    return d

def capitalize_word(s):
    if s.lower() in ['a', 'of', 'to', 'in', 'for', 'on', 'with', 'as', 'by', 'at', 'from', 'and']:
        return s
    return s[:1].capitalize() + s[1:]

def capitalize(s):
    if '.' in s:
        s = '.'.join(map(capitalize_word,s.split('.')[:-1])) + \
            '.' + s.split('.')[-1]
    for delim in [' ', '_', '-', ':']:
        s = delim.join(map(capitalize_word,s.split(delim)))
    return s[:1].capitalize() + s[1:]

def factor(n):
    tmp = n
    factors = []
    for p in [2]+range(3,int(math.sqrt(n)+1),2):
        while tmp % p==0:
            tmp/=p
            factors.append(p)
    if tmp!=1:
        factors.append(tmp)
    return factors

phi_cache = {}
def phi(n):
    if n in phi_cache:
        return phi_cache[n]
    factors = unique(factor(n))
    res = n
    for p in factors:
        res = (p-1)*res/p
    phi_cache[n] = res
    return res

digit2char = lambda d: "0123456789abcdefghijklmnopqrstuvwxyz"[d]

def num2str(n,B = 10):
    digits = []
    if n==0:
        return "0"
    while n>0:
        digits.append(n%B)
        n/=B
    return "".join(map(digit2char,digits[::-1]))

def FourierTransform(f):
    n = math.log(len(f),2)
    if n==0: return f
    h0 = FourierTransform(f[:len(f)/2])
    h1 = FourierTransform(f[len(f)/2:])
    return map(op.add,h0,h1) + map(op.sub,h0,h1)

def FourierCoef(f,func):
    return sum([((1)-2*(poppar(func&i))) * f[i] for i in range(len(f))])


def upto_over(n,k):
    return sum([over(n,i) for i in range(0,k+1)])

def gen_perms(length):
    if length==1:
        yield [0]
    for i in range(length):
        for perm in gen_perms(length-1):
            yield [i] + [j if j<i else j+1 for j in perm]

def gen_perms_seq(lst):
    if len(lst)==1:
        yield [lst[0]]
    for i in range(len(lst)):
        for perm in gen_perms_seq(lst[:i]+lst[i+1:]):
            yield [lst[i]] + perm



log2 = lambda x: log(x,2.0)

def subsets(a,k):
    if len(a)<k:
        return []
    if k==0:
        return [[]]
    res = []
    for i in range(len(a)-k+1):
        x = a[i]
        S = subsets(a[i+1:],k-1)
        for s in S:
            res.append([x]+s[:])
    return res

def prime_sieve(upto):
    N = upto+1
    A = [1]*(N)
    A[0] = A[1] = 0
    for i in range(2,int(math.sqrt(0.1+upto))+1):
        if A[i]:
            for j in range(i,upto/i+1):
                A[i*j] = 0
    return [i for i in range(N) if A[i]]


def prime_sieve2(from_,upto):
    N = upto-from_+1
    A = [1]*(N)
    small_primes = prime_sieve(int(math.sqrt(0.1+upto)))

    for p in small_primes:
        i_min = (from_ - 1)/ p + 1
        i_max = (upto)/p
        for i in range(i_min,i_max+1):
            A[i*p - from_] = 0
    return [i+from_ for i in range(N) if A[i]]

def factor_sieve3(from_,upto, small_primes = None):
    N = upto-from_+1
    A = [[i] for i in range(from_,upto+1)]
    if small_primes is None:
        small_primes = prime_sieve(int(math.sqrt(0.1+upto)))

    for p in small_primes:
        i_min = (from_ - 1)/ p + 1
        i_max = (upto)/p
        for i in range(i_min,i_max+1):
            ind = i*p - from_
            r = A[ind][-1]/p
            A[ind][-1] = p
            j=1
            while r%p == 0:
                A[ind].append(p)
                j+=1
                r/=p
            A[ind].append(r)
    return A


def binary_search(A, from_, to, val):
    """
    A[from_] < val < A[to]
    from_ < val
    """
    if from_ == to-1:
        return from_
    mid = (from_ + to)/2
    if A[mid]==val:
        return mid
    elif A[mid] < val:
        return binary_search(A, mid, to, val)
    else:
        return binary_search(A, from_, mid, val)


def binary_search_fnc(f, from_, to, val, iterations):
    """
    A[from_] < val < A[to]
    from_ < val
    """
    if iterations == 0:
        return (from_ + to)/2
    mid = (from_ + to)/2
    if f(mid) < val:
        return binary_search_fnc(f, mid, to, val, iterations-1)
    else:
        return binary_search_fnc(f, from_, mid, val, iterations-1)


def binary_search2(A, from_, to):
    """
    A[from_] < val < A[to]
    from_ < val
    """
    if from_ == to-1:
        return to
    mid = (from_ + to)/2
    if A(mid) == False:
        return binary_search2(A, mid, to)
    else:
        return binary_search2(A, from_, mid)


def add_to_vec(vec, another):
    for i in range(len(vec)):
        vec[i] += another[i]

def to_num(s):
	return int(''.join(map(str,s)),2)
    
def from_num(x, N):
	return [(x>>(i))&1 for i in range(N-1,-1,-1)]

#from numpy import int32, zeros, ones

def n_factor_sieve(upto):
    N = upto+1
    A = ones((N,),int32)
    P = prime_sieve(upto)
    for i in P:
        A[i] = -1

    for ind in range(len(P)):
        i = P[ind]
        if i > upto/2:
            break
        for j in range(2,upto/i+1):
            A[i*j] *= -1
        for j in range(i,upto/i+1,i):
            A[i*j] = 0
    return A

def factor_sieve(upto):
    N = upto+1
    A = [[] for i in range(N)]
    
    for i in range(2,int(math.sqrt(0.1+upto))+1):
        if len(A[i])==0:
            kmax = int(log(upto, i))
            for k in range(0,kmax+1):
                for j in range(i**k,upto/i+1,i**k):
                    A[i*j].append(i)
    for i in range(2,N):
        m = reduce(lambda x, y: x*y, A[i], 1)
        if m != i:
            A[i].append(i/m)
    return A

def factor_sieve2(upto):
    N = upto+1
    A = [0 for i in range(N)]
    
    for i in range(2,int(math.sqrt(0.1+upto))+1):            
        if A[i]==0:
           for j in range(i, upto/i + 1):
                    A[i*j] = i
    print
    return A

def factor_using_table(FS, n):
    res = []
    while n!=1:
        x=FS[n]
        if x == 0:
            x = n
        res.append(x)
        n /=x
    return res

def gcd_comb(a,b):
    s = 1,0
    t = 0,1
    while b!=0:
        q = a/b
        a, b = b, a - q*b
        s = s[1], s[0] - q*s[1]
        t = t[1], t[0] - q*t[1]
    return s[0],t[0]

def inverse(m,k):
     v = gcd_comb(m,k)
     return (v[0]*m + v[1]*k ==1)*(v[1] % m)

def crt(mods, ps):
    M = product(ps)
    Ms = [M/p for p in ps]
    ys = [inverse(M/p, p) % p for p in ps]
    return sum([ai*Mi*yi for ai,Mi,yi in zip(mods,Ms,ys)]) % M

crt_cache = {}
def fast_crt_prep(ps):
    for i in range(len(ps)):
        p = ps[i]
        for j in range(i+1,len(ps)):
            q = ps[j]
            a,b = gcd_comb(p,q)
            # a*p + b*q = 1 
            crt_cache[p,q] = a % q 
            crt_cache[q,p] = b % p 

def fast_crt(mods, ps):
    M = product(ps)
    Ms = [M/p for p in ps]
    ys = [reduce(lambda x, y: (x*y)% p, [crt_cache[q,p] for q in ps if p!=q]) for p in ps]
    return sum([ai*Mi*yi for ai,Mi,yi in zip(mods,Ms,ys)]) % M

def is_prime(n):
    if n<=1:
        return False
    if n==2:
        return True
    if n%2==0:
        return False
    for p in range(3,int(math.sqrt(n)+1),2):
        if n % p==0:
            return False
        
    return True

def miller_rabin(n,tries = 20):
    # n-1 = 2**r * s
    s = n-1
    r = 0
    while s%2==0:
        s = s/2
        r += 1
    
    for i in range(tries):
        x = random.randint(1,n-1)
        if gcd(x, n)!=1:
            return False
        x = pow(x, s, n)
        if x == 1:
            continue
        for t in range(r):
            x2 = (x*x) % n
            if x2==1:
                if x!=n-1:
                    return False
                else:
                    x = 1
                    break
            x = x2
        if x!=1:
            return False
    return True

def pow(a,x, n):
    if x<1:
        return 1
    if x==1:
        return a%n
    tmp = pow(a,x/2, n)
    tmp = (tmp * tmp)%n
    if x%2==1:
        tmp = (tmp * a)%n
    return tmp

def pow_iter(a,x,n):
    B = bits(x,n)
    r = 1
    for i in range(len(B)):
        r = (r**2)%n
        if B[i]==1:
            r = (r*a)%n
    return r
            
            

def pollard_rho(n, k):
    factors = []
    x1 = x2 = random.randint(0,n-1)
    i = 2
    while i<k:
        
            if n% i==0:
               factors.append(i)
               n = n/i
            else:
                i += 1
            
            
    for i in range(k):
        x1 = (x1**2 + 1)% n
        x2 = (x2**2 + 1)% n
        x2 = (x2**2 + 1)% n
        g = gcd(abs(x1-x2),n)
        if g!=1:
            n = n/g
            factors.append(g)
            if miller_rabin(n):
                break
                
            x1 = x2 = random.randint(0,n-1)
    if n!=1:
        factors.append(n)
    return factors


product = lambda l: reduce(lambda x,y: x*y,l,1)

def sum_divisors(n):
    """
    Sum of divisors
    """
    C = count(factor(n))
    return product([sum([k**i for i in range(v+1)]) for k,v in C.iteritems()])-n

def D2(n):
    """
    Sum of squares of divisors
    """
    C = count(factor(n))
    return product([sum([k**(2*i) for i in range(v+1)]) for k,v in C.iteritems()])

def divisor_count(n):
    """
    Numbers of divisors
    """
    C = count(factor(n))
    return product([(v+1) for k,v in C.iteritems()])-1

def gcd(a,b):
    a,b = max(a,b), min(a,b)
    while b!=0:
        a,b = b, a%b
    return a

def partitions(seq):
    if len(seq)==1:
        return [[seq[:]]]
    if len(seq)==0:
        return [[]]
    res = []
    for i in range(1<<(len(seq)-1)):
        nseq = [seq[0]] + [seq[j] for j in range(1,len(seq)) if ((i>>(j-1))&1==1)]
        leftseq = [seq[j] for j in range(1,len(seq)) if ((i>>(j-1))&1==0)]
        P = partitions(leftseq)
        for p in P:
            res.append([nseq] + p)
    return res

def par(n, m=1):
    if n==0:
        return [[]]
    if m>n:
        return []
    res = []
    for i in range(m,n+1):
        t = par(n-i,i)
        for x in t:
            res.append([i]+x)
    return res
        
def h_exp(a,b,m):
    if m==1:
        return 0

    if b==1:
        return a

    t = h_exp(a,b-1,phi(m))
    return pow(a, t, m)

def search(st, sub):
    pos = -1
    res = []
    while True:
        pos = st.find(sub,pos + 1)
        if pos==-1:
            break
        res.append(pos)
    return res

def cartesian(seqs):
    if len(seqs)==1:
        return [[x] for x in seqs[0]]
    c = cartesian(seqs[1:])
    res = []
    for x in seqs[0]:
        for y in c:
            res.append([x]+y)
    return res



def drange(start, stop, step):
	p = start
	while p<=stop+0.00000001:
		yield p
		p+= step

def group_by(func, vals):
    D = {}
    for v in vals:
        k = func(v)
        D[k] = D.get(k,[])+[v]
    return D

# First argument what you got, second argument the distribution you started from.
Chernoff = lambda x,y,base=math.e: x*math.log(x/y,base)  + (1-x) * math.log((1-x)/(1-y),base)

def search_min(f, from_, to, iter = 50):
    for i in range(iter):
        points = [from_ + ((to-from_)*i/4.) for i in range(5)]
        vals = [f(p) for p in points]
        amin = argmin(vals)
        if amin==0:
            amin = 1
        elif amin==4:
            amin = 3
        from_ = points[amin-1]
        to = points[amin+1]
    point = (from_ + to)/2
    return point, f(point)

ent = lambda p: (0 if p==0 else -(p*log2(p)+(1-p)*log2(1-p)))
entinv = lambda x: search_zero(lambda p: ent(p)-x,0.,0.5)

def search_zero(f, from_, to, iter = 50):
    for i in range(iter):
        middle = (from_ + to)/2
        val = f(middle)
        if sign(val)==sign(f(from_)):
            from_ = middle
        else:
            to = middle
    return middle
    

def all_subsets(S):
    res = []
    for i in range(1<<len(S)):
        t = bits(i,len(S))
        res.append([S[j] for j in range(len(S)) if t[j]])
    return res

##FS=factor_sieve2(10**6)
##for i in range(2,1000):
##    L = factor_using_table(FS,i)*2
##    S = sum_list([[x]*(L.count(x)/3)for x in set(L) if L.count(x)>=3])
##    map(product,all_subsets(S))
##    print i,S



def tensor_product(A,B):
    n1, m1 = len(A),len(A[0])
    n2, m2 = len(B),len(B[0])
    C = [[0]*(m1*m2) for i in range(n1*n2)]
    for i in range(n1*n2):
        i1,i2 = i%n1,i/n1
        for j in range(m1*m2):
            j1,j2 = j%m1,j/m1
            C[i][j] = A[i1][j1] * B[i2][j2]
    return C



def time_it(st):
    t = time.time(); ret = eval(st); print(time.time()-t)
    return ret


def tqdm(lst, jump = 48):
    tt = 0.
    i = 0
    timestamp = time.time()
    lasttime = timestamp
    for x in lst:
        yield x
        if i%jump==0:
            print(i, "%.4f" % (time.time() - timestamp))
        i+=1
    print
