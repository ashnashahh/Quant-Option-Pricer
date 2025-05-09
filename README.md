# European Option Pricer (Python)

Two compact engines for pricing **European call/put options**:

* **Black-Scholes analytic** formula  
* **Vectorised Monte-Carlo** simulation (NumPy)

Built as a coding sample for quantitative-finance graduate applications  
 and interview take-homes.

---

## Quick start

```bash
# Install deps
pip install -r requirements.txt

# Closed-form price
python option_pricer.py --model bs --S 100 --K 105 --r 0.03 --sigma 0.25 --T 0.5 --type call

# Monte-Carlo price (200K paths)
python option_pricer.py --model mc --S 100 --K 105 --r 0.03 --sigma 0.25 --T 0.5 --type put --paths 200000
---

### Author

**Ashna Shah**  
GitHub · [@ashna-shah](https://github.com/ashna-shah) | LinkedIn · [@ashna-shah](https://linkedin.com/in/ashna-shah)  
Email: shahashna05@gmail.com
