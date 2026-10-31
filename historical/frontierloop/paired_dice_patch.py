"""Explicit, opt-in initializer adapter for pinned GPL-3.0 DICE source.

Only initial determinant input/validation/audit changes. This is NOT an
unmodified upstream binary, a new HCI algorithm, or a new SCI theory.
"""
ENV='CEO_DICE_PAIRED_INPUT_V1'
ORIGINAL_SHA256='b11f00a73737624fa14a8e16efbc62790988d6eb362833b0b67620a3c15909f5'

ENVIRONMENT='''  const char* ceo_pair_env = std::getenv("CEO_DICE_PAIRED_INPUT_V1");
  if (ceo_pair_env && std::string(ceo_pair_env) != "1") {
    std::cerr << "CEO_PAIRED_INPUT_INVALID_ENV" << std::endl;
    return 2;
  }
  const bool ceo_paired_input = ceo_pair_env != nullptr;
'''
ALPHA_GUARD='''    if (ceo_paired_input && (buffer_a.empty() || buffer_a.size()%16 ||
        buffer_a.size()/16 > 32768 || norbs/2 > 32)) {
      std::cerr << "CEO_PAIRED_INPUT_INVALID_ALPHA" << std::endl; return 2;
    }
    auto ceo_valid_strings = [&](const std::vector<unsigned char>& bytes) {
      int expected = -1;
      for (size_t i=0; i<bytes.size()/16; ++i) {
        int count=0;
        for (int c=0; c<16; ++c) for (int b=0; b<8; ++b) {
          if ((bytes[16*i+15-c]>>b)&1) {
            if (c*8+b >= norbs/2) return false;
            ++count;
          }
        }
        if (expected == -1) expected=count;
        if (count != expected) return false;
      }
      return expected >= 0;
    };
    if (ceo_paired_input && !ceo_valid_strings(buffer_a)) {
      std::cerr << "CEO_PAIRED_INPUT_ALPHA_SECTOR" << std::endl; return 2;
    }
'''
BETA_GUARD='''    if (ceo_paired_input && (buffer_b.size()!=buffer_a.size() ||
        !ceo_valid_strings(buffer_b))) {
      std::cerr << "CEO_PAIRED_INPUT_INVALID_BETA" << std::endl; return 2;
    }
'''
AUDIT='''    if (ceo_paired_input) {
      std::set<std::pair<unsigned long long,unsigned long long>> seen;
      std::ofstream audit("CEO-initial-pairs-v1.txt");
      if (!audit) { std::cerr << "CEO_PAIRED_INPUT_NO_AUDIT" << std::endl; return 2; }
      for (int i=0; i<nAlphaDets; ++i) {
        unsigned long long a=0,b=0;
        for (int k=0; k<nalpha; ++k) a |= 1ULL << occAlpha[i][k];
        for (int k=0; k<nbeta; ++k) b |= 1ULL << occBeta[i][k];
        if (!seen.insert(std::make_pair(a,b)).second) {
          std::cerr << "CEO_PAIRED_INPUT_DUPLICATE_PAIR" << std::endl; return 2;
        }
        audit << a << " " << b << "\\n";
      }
      audit.close();
      if (!audit) { std::cerr << "CEO_PAIRED_INPUT_AUDIT_WRITE_FAILED" << std::endl; return 2; }
    }
'''


def apply(source):
    replacements=[
        ('  int lowestEnergyDet = 0;\n', ENVIRONMENT+'  int lowestEnergyDet = 0;\n'),
        ('    int nAlphaDets = static_cast<int>(buffer_a.size()/detsize);\n',
         ALPHA_GUARD+'    int nAlphaDets = static_cast<int>(buffer_a.size()/detsize);\n'),
        ('    int nBetaDets = static_cast<int>(buffer_b.size()/detsize);\n',
         BETA_GUARD+'    int nBetaDets = static_cast<int>(buffer_b.size()/detsize);\n'),
        ('    // Make HF determinant\n    Dets.resize(nAlphaDets*nBetaDets);\n',
         AUDIT+'    // Make initial determinants: optional aligned pairs, otherwise original product.\n'
         '    Dets.resize(ceo_paired_input ? nAlphaDets : nAlphaDets*nBetaDets);\n'),
        ('      for (int j=0; j<nBetaDets; j++) {\n        Determinant& d = Dets[i*nBetaDets+j];',
         '      for (int j=(ceo_paired_input ? i : 0); j<(ceo_paired_input ? i+1 : nBetaDets); j++) {\n'
         '        const int ceo_idx = ceo_paired_input ? i : i*nBetaDets+j;\n'
         '        Determinant& d = Dets[ceo_idx];'),
        ('          lowestEnergyDet = i*nBetaDets+j;', '          lowestEnergyDet = ceo_idx;')]
    for before,after in replacements:
        if source.count(before)!=1:raise ValueError('Pinned source context missing/ambiguous; no fuzzy patch')
        source=source.replace(before,after)
    return source
