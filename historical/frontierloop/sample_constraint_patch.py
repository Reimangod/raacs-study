"""Input-pool constraint for pinned GPL DICE variational candidate selection.

This reproduces an admissible-universe restriction, not the complete published
HCI-HSQD workflow. The unset mode retains native unrestricted selection.
"""
ORIGINAL_SHA256='115aa0f8fdf8d066b3ba69ae34e698ca9516398dd006b69fef155f2f7589c98c'
HELPERS=r'''
// Modified research adapter: optional observed-half-pool constraint, 2026-10-26.
// Original GPL DICE Hamiltonian and selection routines are retained.
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <set>
#include <vector>
#include <algorithm>
namespace {
struct CeoSamplePool {
  bool enabled=false;
  int norb=0;
  std::set<std::uint64_t> strings;
};
void ceo_sample_fail(const char* reason) {
  std::cerr << "CEO_SAMPLE_POOL_INVALID_" << reason << std::endl;
  std::exit(2);
}
CeoSamplePool ceo_sample_load(int nelec) {
  CeoSamplePool p;
  const char* mode=std::getenv("CEO_DICE_SAMPLE_POOL_V1");
  if (!mode) return p;
  if (std::string(mode)!="1") ceo_sample_fail("MODE");
  p.enabled=true; p.norb=Determinant::norbs/2;
  if (p.norb<1 || p.norb>32 || nelec<2 || nelec%2 || nelec/2>p.norb)
    ceo_sample_fail("SECTOR");
  std::ifstream in("AllowedHalfDets.bin",std::ios::binary);
  if (!in) ceo_sample_fail("MISSING");
  in.seekg(0,std::ios::end);const std::streamoff size=in.tellg();
  if (size<16 || size%16 || size>1600000) ceo_sample_fail("LENGTH");
  in.seekg(0);std::vector<unsigned char> bytes(static_cast<size_t>(size));
  in.read(reinterpret_cast<char*>(bytes.data()),size);
  if (!in) ceo_sample_fail("READ");
  for (size_t i=0;i<bytes.size();i+=16) {
    std::uint64_t value=0;
    for (int j=0;j<8;++j) if (bytes[i+j]) ceo_sample_fail("HIGH_BITS");
    for (int j=8;j<16;++j) value=(value<<8)|bytes[i+j];
    if ((value>>p.norb) || __builtin_popcountll(value)!=nelec/2)
      ceo_sample_fail("BITS_OR_SECTOR");
    if (!p.strings.insert(value).second) ceo_sample_fail("DUPLICATE");
  }
  const std::uint64_t hf=(1ULL<<(nelec/2))-1;
  if (!p.strings.count(hf)) ceo_sample_fail("NO_HF");
  std::ofstream audit("CEO-sample-pool-v1.txt");
  if (!audit) ceo_sample_fail("AUDIT");
  for (auto x:p.strings) audit << x << "\n";
  audit.close();if (!audit) ceo_sample_fail("AUDIT_WRITE");
  return p;
}
bool ceo_sample_member(Determinant d,const CeoSamplePool& p) {
  std::uint64_t a=0,b=0;
  for (int i=0;i<p.norb;++i) {
    if (d.getocc(2*i)) a|=1ULL<<i;
    if (d.getocc(2*i+1)) b|=1ULL<<i;
  }
  return p.strings.count(a) && p.strings.count(b);
}
}

'''
INITIAL=r'''
  const auto ceo_sample_pool=ceo_sample_load(nelec);
  if (ceo_sample_pool.enabled) {
    for (auto& x:Dets) if (!ceo_sample_member(x,ceo_sample_pool))
      ceo_sample_fail("INITIAL_OUTSIDE_POOL");
  }
'''
FILTER=r'''
    if (ceo_sample_pool.enabled) {
      const auto generated=uniqueDEH.Det->size();
      uniqueDEH.Det->erase(std::remove_if(uniqueDEH.Det->begin(),uniqueDEH.Det->end(),
          [&](const Determinant& x) {return !ceo_sample_member(x,ceo_sample_pool);}),
          uniqueDEH.Det->end());
      pout << "CEO_SAMPLE_POOL_V1 " << iter << " " << generated << " "
           << uniqueDEH.Det->size() << std::endl;
    }
'''


def apply(source):
    if 'struct CeoSamplePool' in source:raise ValueError('Sample-pool patch already applied')
    replacements=[
        ('vector<double> SHCIbasics::DoVariational(vector<MatrixXx> &ci,',
         HELPERS+'vector<double> SHCIbasics::DoVariational(vector<MatrixXx> &ci,'),
        ('  int SortedDetsSize = SortedDetsvec.size(), DetsSize = Dets.size();\n',
         '  int SortedDetsSize = SortedDetsvec.size(), DetsSize = Dets.size();\n'+INITIAL),
        ('    // Remove duplicates and put all the dets on all the nodes\n',
         FILTER+'    // Remove duplicates and put all the dets on all the nodes\n')]
    for before,after in replacements:
        if source.count(before)!=1:raise ValueError('Pinned source context drift; no fuzzy patch')
        source=source.replace(before,after)
    return source
