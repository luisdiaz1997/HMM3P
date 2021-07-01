import bioframe as bf



def arm_regions(assembly, chromsizes):
    midpoints = bf.fetch_centromeres(assembly)
    chromarms = bf.make_chromarms(chromsizes, midpoints )
    chromarms = chromarms.iloc[ np.array([i.isdigit() for i in chromarms.chrom.str.replace('chr','').values])]
    chromarms = chromarms.iloc[ np.array([i.isdigit() for i in chromarms.chrom.str.replace('chr','').values])]
    chromarms = chromarms.sort_values(by=['chrom', 'start'],  key=lambda col: col.astype(str).str.replace('chr','').astype(int))
    regions = [(row[0], row[1], row[2]) for index, row in chromarms.iterrows() if (row[0].replace('chr','')).isdigit()]
    return regions

def chrom_regions(assembly, chromsizes):
    regions = [(chrom, 0, chromsizes[chrom]) for chrom in chromsizes.index if (chrom.replace('chr','')).isdigit() ]
    return regions