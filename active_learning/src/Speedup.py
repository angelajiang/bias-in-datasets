def get_percent_speedup(nb0, nb1, nf1, alpha, kath_selectivity=None):
    '''
    nb0: num backprops original
    nb1: num backprops new
    nf1: num forwards new
    alpha: Latency_backwards / Latency_forwards
    '''

    if kath_selectivity > 0:
        nf1 = nb1 * kath_selectivity

    Lf = 1.
    Lb = alpha
    latency_original = (Lb * nb0) + (Lf * nb0)
    latency_new = (Lb * nb1) + (Lf * nf1)
    #return (latency_original - latency_new) * 100. / latency_original
    return float(latency_original) / latency_new



