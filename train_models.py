"""
Banking Semantic Search — Dataset Generation & Model Training Pipeline

Generates a synthetic banking customer-support dataset (~1000 queries across 5 categories),
preprocesses it, creates stratified train/val/test splits, and trains both
Word2Vec and FastText models.

Usage:
    python train_models.py
"""

import os
import re
import random
import string
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec, FastText

# ---------------------------------------------------------------------------
# 0. NLTK data
# ---------------------------------------------------------------------------
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# 1. Synthetic dataset generation
# ---------------------------------------------------------------------------

QUERIES = {
    "Account Management": [
        "How do I open a new savings account?",
        "What are the requirements to open a checking account?",
        "How can I close my bank account?",
        "What is the minimum balance for a savings account?",
        "How do I update my contact information?",
        "Can I change my account type from savings to checking?",
        "How do I set up direct deposit?",
        "What are the different types of accounts you offer?",
        "How can I check my account balance online?",
        "How do I link my accounts together?",
        "What is the interest rate on my savings account?",
        "How do I add a joint account holder?",
        "Can I remove a joint account holder?",
        "How do I request a new debit card?",
        "What is the daily withdrawal limit?",
        "How can I increase my daily withdrawal limit?",
        "How do I set up automatic transfers between accounts?",
        "Can I have multiple savings accounts?",
        "How do I view my account statements?",
        "How do I download my bank statements in PDF?",
        "What are the fees for maintaining an account?",
        "How do I set up paperless statements?",
        "Can I open an account for my child?",
        "What documents do I need to open an account?",
        "How do I change my mailing address?",
        "How can I update my email address on file?",
        "How do I change my phone number on my account?",
        "What is a money market account?",
        "How does a certificate of deposit work?",
        "What are the penalties for early CD withdrawal?",
        "How do I transfer my account to another branch?",
        "Can I access my account from another country?",
        "How do I set up account alerts?",
        "What notifications can I receive for my account?",
        "How do I opt out of marketing communications?",
        "Can I set a custom nickname for my accounts?",
        "How do I reorder checks?",
        "What is the routing number for my account?",
        "How do I find my account number?",
        "Can I merge two accounts into one?",
        "How do I switch my primary account?",
        "What are overdraft protection options?",
        "How do I sign up for overdraft protection?",
        "How do I disable overdraft protection?",
        "What happens if my account goes negative?",
        "How long does it take to open a new account?",
        "Can I open an account online?",
        "Do you offer student bank accounts?",
        "What benefits come with a premium account?",
        "How do I upgrade to a premium account?",
        "Can I downgrade my account type?",
        "What is the annual percentage yield on savings?",
        "How do I set a spending limit on my account?",
        "Can I restrict certain transactions on my account?",
        "How do I unfreeze my account?",
        "Why was my account frozen?",
        "How do I reactivate a dormant account?",
        "What happens to a dormant account?",
        "How can I recover funds from a closed account?",
        "How do I update my signature on file?",
        "Can I get a notarized letter from the bank?",
        "How do I request an account verification letter?",
        "What is the process for estate accounts?",
        "How do I set up a trust account?",
        "What are the tax implications of my savings interest?",
        "How do I get my 1099-INT form?",
        "Can I deposit checks using my phone?",
        "How does mobile check deposit work?",
        "What is the mobile deposit limit?",
        "How long does a mobile deposit take to clear?",
        "Can I deposit cash at an ATM?",
        "How do I find the nearest ATM?",
        "Are there fees for using out-of-network ATMs?",
        "How do I set up a recurring deposit?",
        "What are your hours of operation?",
        "How do I schedule an appointment with a banker?",
        "Can I get a safe deposit box?",
        "What are the fees for a safe deposit box?",
        "How do I reset my online banking password?",
        "I forgot my online banking username",
        "How do I enable two-factor authentication?",
        "How do I register for online banking?",
        "Can I manage my account through the mobile app?",
        "How do I download the mobile banking app?",
        "Is mobile banking secure?",
        "How do I log out of mobile banking?",
        "Why can't I log in to my account?",
        "My account is locked, how do I unlock it?",
        "How do I change my PIN number?",
        "Can I set up Face ID for mobile banking?",
        "How do I enable fingerprint login?",
        "What browsers are supported for online banking?",
        "How do I clear my online banking cache?",
        "Can I access online banking on multiple devices?",
        "How do I update the mobile banking app?",
        "What features are available in the mobile app?",
        "How do I provide feedback about banking services?",
        "How do I file a complaint with the bank?",
    ],
    "Loans": [
        "How do I apply for a personal loan?",
        "What is the interest rate on a home loan?",
        "What are the eligibility requirements for a mortgage?",
        "How long does loan approval take?",
        "What documents do I need for a loan application?",
        "How do I check my loan application status?",
        "What is the maximum loan amount I can get?",
        "How do I calculate my monthly loan payment?",
        "What are the different types of home loans?",
        "Can I get a loan with bad credit?",
        "What is a fixed-rate mortgage?",
        "What is an adjustable-rate mortgage?",
        "How does refinancing work?",
        "When should I refinance my mortgage?",
        "What are the closing costs for a mortgage?",
        "How do I get pre-approved for a home loan?",
        "What is the difference between pre-qualified and pre-approved?",
        "How do I apply for an auto loan?",
        "What are the current auto loan rates?",
        "Can I refinance my car loan?",
        "How do I make extra payments on my loan?",
        "Is there a penalty for paying off my loan early?",
        "How do I set up automatic loan payments?",
        "What happens if I miss a loan payment?",
        "How do I defer my loan payment?",
        "Can I modify my loan terms?",
        "How do I apply for a home equity loan?",
        "What is a home equity line of credit?",
        "What is the difference between HELOC and home equity loan?",
        "How much equity do I need for a HELOC?",
        "What are the tax benefits of a home loan?",
        "How do I get a construction loan?",
        "What is a bridge loan?",
        "How do I apply for a student loan?",
        "What are the student loan repayment options?",
        "Can I consolidate my student loans?",
        "How does student loan forgiveness work?",
        "What is a co-signer and do I need one?",
        "How does my credit score affect my loan rate?",
        "What credit score do I need for a mortgage?",
        "How do I improve my chances of loan approval?",
        "What is PMI and when is it required?",
        "How do I remove PMI from my mortgage?",
        "What is an FHA loan?",
        "What is a VA loan and who qualifies?",
        "What is a USDA loan?",
        "How do I apply for a small business loan?",
        "What collateral is needed for a business loan?",
        "How do I get an SBA loan?",
        "What are the terms for a line of credit?",
        "How do I increase my line of credit?",
        "What is a debt consolidation loan?",
        "Should I consolidate my debts?",
        "How do I calculate my debt-to-income ratio?",
        "What is the loan-to-value ratio?",
        "How do I request a loan payoff statement?",
        "What is an amortization schedule?",
        "How do I read my amortization schedule?",
        "Can I transfer my loan to another person?",
        "What is loan assumption?",
        "How do I apply for a second mortgage?",
        "What are the risks of a second mortgage?",
        "How do I get a loan for home improvements?",
        "What is a personal line of credit?",
        "How does a secured loan differ from unsecured?",
        "What is the prime rate?",
        "How does the prime rate affect my loan?",
        "Can I change from variable to fixed rate?",
        "What is a balloon payment?",
        "How do I negotiate a better loan rate?",
        "What is the grace period for loan payments?",
        "How do I set up biweekly loan payments?",
        "What is an escrow account?",
        "How does my escrow account work?",
        "Why did my mortgage payment increase?",
        "How do I appeal a loan denial?",
        "What is a jumbo loan?",
        "How do I get a loan for an investment property?",
        "What are the current mortgage rates?",
        "How do I lock in my mortgage rate?",
        "What is a rate lock?",
        "How long does a rate lock last?",
        "Can I get a construction-to-permanent loan?",
        "What is the process for a loan modification?",
        "How do I apply for forbearance?",
        "What is the difference between forbearance and deferment?",
        "How do I get a hardship loan?",
        "What are payday loan alternatives?",
        "Can I pay my loan with a credit card?",
        "What is the total cost of my loan over its lifetime?",
        "How do I track my loan payments?",
        "What happens at the end of my loan term?",
        "How do I get my lien released after payoff?",
        "What is a title search?",
    ],
    "Card Services": [
        "How do I apply for a credit card?",
        "What credit cards do you offer?",
        "How do I activate my new credit card?",
        "How do I report a lost credit card?",
        "How do I block my lost credit card?",
        "How do I request a replacement card?",
        "What is my credit card limit?",
        "How do I increase my credit card limit?",
        "How do I decrease my credit card limit?",
        "What are the fees for my credit card?",
        "How do I view my credit card statement?",
        "When is my credit card payment due?",
        "How do I make a credit card payment?",
        "What is the minimum payment on my credit card?",
        "How do I set up autopay for my credit card?",
        "How do I earn rewards on my credit card?",
        "How do I redeem my credit card rewards?",
        "What is my current rewards balance?",
        "Do my rewards points expire?",
        "Can I transfer my rewards to another program?",
        "How do I dispute a charge on my credit card?",
        "I see an unauthorized charge on my card",
        "How long does a dispute take to resolve?",
        "What is a chargeback?",
        "How do I request a chargeback?",
        "What is the APR on my credit card?",
        "How is credit card interest calculated?",
        "What is a balance transfer?",
        "How do I do a balance transfer?",
        "What are the balance transfer fees?",
        "How do I get a cash advance on my credit card?",
        "What are cash advance fees?",
        "What is the foreign transaction fee on my card?",
        "Does my card work internationally?",
        "How do I set up travel notifications for my card?",
        "Why was my card declined?",
        "How do I unblock my credit card?",
        "How do I add my card to a digital wallet?",
        "Can I use my card with Apple Pay?",
        "Can I use my card with Google Pay?",
        "How do I set up contactless payments?",
        "How do I request a virtual card number?",
        "What is a secured credit card?",
        "How do I upgrade from a secured to unsecured card?",
        "What is a business credit card?",
        "How do I apply for a business credit card?",
        "Can I add an authorized user to my card?",
        "How do I remove an authorized user?",
        "What spending limits can I set for authorized users?",
        "How do I cancel my credit card?",
        "What happens to my rewards if I cancel?",
        "Will canceling my card affect my credit score?",
        "How do I freeze my credit card temporarily?",
        "How do I unfreeze my credit card?",
        "What purchase protection does my card offer?",
        "Does my card have travel insurance?",
        "What is extended warranty protection?",
        "How do I file an insurance claim through my card?",
        "What is the grace period on my credit card?",
        "How do I avoid paying interest on my credit card?",
        "What is a credit card promotional rate?",
        "When does my promotional rate expire?",
        "How do I apply for a student credit card?",
        "What is a co-branded credit card?",
        "How do I set spending alerts on my card?",
        "Can I customize the design of my card?",
        "How do I get a metal credit card?",
        "What are the perks of a premium credit card?",
        "How do I apply for a platinum card?",
        "What is the annual fee for my card?",
        "Can I get the annual fee waived?",
        "How does my credit card billing cycle work?",
        "What is a statement closing date?",
        "How do I change my credit card PIN?",
        "Can I get cash back at the point of sale?",
        "What is chip and PIN technology?",
        "How do I report a damaged card?",
        "Why did I receive a new card I didn't request?",
        "How do I update my card information with merchants?",
        "What is a recurring charge and how do I manage it?",
        "How do I stop a recurring charge on my card?",
        "Can I set a daily spending limit on my card?",
        "How do I track my credit card spending?",
        "What tools do you offer for credit card management?",
        "How do I view my credit score through the bank?",
        "Does checking my credit score affect it?",
        "How often is my credit score updated?",
        "What factors affect my credit score?",
        "How do I improve my credit score?",
        "Can I get a credit card with no annual fee?",
        "What is the difference between Visa and Mastercard?",
        "How do I choose the right credit card for me?",
    ],
    "Payments & Transfers": [
        "How do I send money to another account?",
        "How do I transfer money between my accounts?",
        "How do I set up a wire transfer?",
        "What are the fees for wire transfers?",
        "How long does a wire transfer take?",
        "How do I send an international wire transfer?",
        "What information do I need for an international transfer?",
        "What is a SWIFT code?",
        "How do I find the SWIFT code for my bank?",
        "What is an IBAN number?",
        "How do I set up recurring payments?",
        "How do I cancel a recurring payment?",
        "How do I modify a scheduled payment?",
        "How do I pay my bills online?",
        "How do I set up bill pay?",
        "Can I schedule future-dated payments?",
        "What is the cutoff time for same-day transfers?",
        "How do I send money via Zelle?",
        "How do I receive money through Zelle?",
        "Is Zelle free to use?",
        "What are the limits for Zelle transfers?",
        "How do I send money to someone without a bank account?",
        "What is a cashier's check?",
        "How do I get a cashier's check?",
        "What is the cost of a cashier's check?",
        "How do I send a money order?",
        "Where can I cash a money order?",
        "How do I stop a payment?",
        "Can I reverse a bank transfer?",
        "How do I cancel a pending transaction?",
        "What is ACH transfer?",
        "How long does an ACH transfer take?",
        "What is the difference between ACH and wire transfer?",
        "How do I set up direct deposit for my paycheck?",
        "How do I change my direct deposit information?",
        "How do I verify a direct deposit was received?",
        "How do I pay someone with my phone?",
        "What peer-to-peer payment options do you offer?",
        "How do I request money from someone?",
        "How do I split a payment with friends?",
        "What is a standing order?",
        "How do I set up a standing order?",
        "How do I cancel a standing order?",
        "What is the difference between a standing order and direct debit?",
        "How do I set up a direct debit?",
        "How do I cancel a direct debit?",
        "What happens if a payment fails?",
        "How do I handle a returned payment?",
        "What is the hold time for deposited checks?",
        "How do I check the status of a transfer?",
        "How do I view my transaction history?",
        "How do I search for a specific transaction?",
        "How do I export my transaction history?",
        "Can I categorize my transactions?",
        "How do I set up payment reminders?",
        "What is an e-transfer?",
        "How do I receive an e-transfer?",
        "How long does an e-transfer take?",
        "What are the limits for e-transfers?",
        "Is there a fee for e-transfers?",
        "How do I make a payment using QR code?",
        "Can I pay internationally using my bank app?",
        "What exchange rates do you offer?",
        "How do I lock in an exchange rate?",
        "What are the foreign exchange fees?",
        "How do I buy foreign currency?",
        "Can I hold multiple currencies in my account?",
        "How do I make a tax payment through the bank?",
        "How do I set up payroll through my business account?",
        "What is a batch payment?",
        "How do I process batch payments?",
        "What is real-time payment?",
        "How do I enable real-time payments?",
        "How do I send money to a charity?",
        "Can I schedule automatic charitable donations?",
        "How do I set up automatic savings transfers?",
        "What is a sweep account?",
        "How do I resolve a duplicate payment?",
        "What is a payment gateway?",
        "How do I integrate payment processing for my business?",
        "Can I receive payments through my bank?",
        "How do I set up invoicing through the bank?",
        "What is contactless payment?",
        "How do I use tap to pay?",
        "Can I make payments using my smartwatch?",
        "How do I pay my mortgage online?",
        "How do I pay my loan installment?",
        "How do I pay my credit card bill through the app?",
        "What is auto-debit for loan payments?",
        "How do I set up auto-debit?",
        "Can I pay utility bills through the bank?",
        "How do I verify a payment was received?",
    ],
    "Fraud & Security": [
        "I think someone accessed my account without permission",
        "How do I report a fraudulent transaction?",
        "My debit card was stolen, what should I do?",
        "How do I freeze my account due to suspicious activity?",
        "What should I do if I received a phishing email?",
        "How do I report a phishing text message?",
        "Is this email really from the bank?",
        "How do I verify a phone call is from the bank?",
        "What is two-factor authentication?",
        "How do I enable two-factor authentication?",
        "How do I change my security questions?",
        "What security features does the bank offer?",
        "How do I set up fraud alerts?",
        "What are transaction alerts?",
        "How do I enable real-time transaction alerts?",
        "How do I set spending limits for security?",
        "What do I do if I suspect identity theft?",
        "How do I place a fraud alert on my credit report?",
        "How do I place a credit freeze?",
        "What is the difference between a fraud alert and credit freeze?",
        "How do I report a lost or stolen checkbook?",
        "What happens after I report fraud?",
        "How long does a fraud investigation take?",
        "Will I get my money back after fraud?",
        "What is the bank's liability policy for fraud?",
        "How do I file a police report for bank fraud?",
        "How do I check if my account has been compromised?",
        "What are signs that my account has been hacked?",
        "How do I secure my online banking account?",
        "What password requirements does the bank have?",
        "How often should I change my banking password?",
        "How do I create a strong banking password?",
        "What is biometric authentication?",
        "How do I set up biometric login?",
        "Is it safe to use public WiFi for banking?",
        "How do I protect myself from ATM skimming?",
        "What is card skimming?",
        "How can I tell if an ATM has a skimmer?",
        "What should I do if I gave my information to a scammer?",
        "How do I report a suspicious website?",
        "What is social engineering fraud?",
        "How do I protect myself from social engineering?",
        "What are common banking scams?",
        "How do I recognize a banking scam?",
        "What is account takeover fraud?",
        "How do I prevent account takeover?",
        "What is SIM swap fraud?",
        "How do I protect against SIM swap attacks?",
        "How do I report unauthorized wire transfers?",
        "What is the bank's data encryption standard?",
        "How does the bank protect my personal information?",
        "What is the bank's privacy policy?",
        "How do I opt out of data sharing?",
        "Can I see who has accessed my account information?",
        "How do I report a data breach?",
        "What should I do after a data breach notification?",
        "How do I dispute a transaction I didn't make?",
        "What documentation do I need to file a fraud claim?",
        "How do I get a new account number after fraud?",
        "What is check fraud?",
        "How do I report check fraud?",
        "What is elder financial abuse?",
        "How do I report suspected elder financial abuse?",
        "What protections exist for vulnerable customers?",
        "How do I set up trusted contact for my account?",
        "What is a security token?",
        "How do I use a security token for login?",
        "How do I report unauthorized account changes?",
        "What is device registration for security?",
        "How do I manage my registered devices?",
        "How do I remove a device from my account?",
        "What happens if I enter wrong password too many times?",
        "How do I recover my account after lockout?",
        "Is the mobile app safe to use?",
        "How do I report a vulnerability in the bank's system?",
    ],
}


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    """Build the synthetic dataset from the query templates."""
    random.seed(seed)
    rows = []
    uid = 1
    for category, queries in QUERIES.items():
        for q in queries:
            rows.append({"id": uid, "query": q, "category": category})
            uid += 1
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Cleaning functions
# ---------------------------------------------------------------------------

def light_clean(text: str) -> str:
    """Minimal cleaning — collapse whitespace, strip non-ASCII."""
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deep_clean(text: str) -> str:
    """Aggressive cleaning — lowercase, remove punctuation, stopwords, lemmatize."""
    text = light_clean(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 3. Tokenizer (must be identical to search_engine.py)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase → strip non-letter chars → word_tokenize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return word_tokenize(text)


# ---------------------------------------------------------------------------
# 4. Sentence vectorizer
# ---------------------------------------------------------------------------

def get_sentence_vector(model, tokens: list[str]) -> np.ndarray:
    """Mean-pool word vectors. FastText handles OOV via char n-grams."""
    vectors = []
    for token in tokens:
        try:
            vectors.append(model.wv[token])
        except KeyError:
            continue
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# ---------------------------------------------------------------------------
# 5. Main training pipeline
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ---- Generate dataset ----
    print("📋  Generating synthetic dataset …")
    df = generate_dataset()
    print(f"    Total queries: {len(df)}")
    print(f"    Categories: {df['category'].value_counts().to_dict()}")
    df.to_csv(os.path.join(data_dir, "banking_semantic_search_dataset.csv"), index=False)

    # ---- Drop duplicates ----
    df = df.drop_duplicates(subset="query").reset_index(drop=True)

    # ---- Clean ----
    df["query_clean"] = df["query"].apply(light_clean)
    df["query_deep"]  = df["query"].apply(deep_clean)

    # ---- Label-encode categories ----
    categories = sorted(df["category"].unique())
    cat_map = {c: i for i, c in enumerate(categories)}
    df["category_id"] = df["category"].map(cat_map)

    df.to_csv(os.path.join(data_dir, "banking_dataset_clean.csv"), index=False)

    # ---- Stratified split 80/10/10 ----
    print("🔀  Stratified split …")
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["category_id"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["category_id"], random_state=42
    )
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ---- Tokenize training data ----
    train_tokens = [tokenize(q) for q in train_df["query_clean"]]

    # ---- Train Word2Vec ----
    print("🧠  Training Word2Vec …")
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=100, window=5, min_count=1, sg=1, epochs=50, seed=42,
    )
    w2v_path = os.path.join(models_dir, "word2vec_banking.model")
    w2v_model.save(w2v_path)
    print(f"    Saved → {w2v_path}  (vocab: {len(w2v_model.wv)})")

    # ---- Train FastText ----
    print("🧠  Training FastText …")
    ft_model = FastText(
        sentences=train_tokens,
        vector_size=100, window=5, min_count=1, sg=1, epochs=50,
        min_n=3, max_n=6, seed=42,
    )
    ft_path = os.path.join(models_dir, "fasttext_banking.model")
    ft_model.save(ft_path)
    print(f"    Saved → {ft_path}  (vocab: {len(ft_model.wv)})")

    # ---- Build training reference vectors (FastText) ----
    print("📐  Building training reference vectors …")
    train_vectors = np.array([
        get_sentence_vector(ft_model, tokenize(q))
        for q in train_df["query_clean"]
    ])
    np.save(os.path.join(models_dir, "train_vectors_fasttext.npy"), train_vectors)

    # Save a copy of train.csv in models/ for the search engine
    train_df[["id", "query", "query_clean", "category", "category_id"]].to_csv(
        os.path.join(models_dir, "train.csv"), index=False
    )

    # ---- Quick validation: top-1 NN accuracy on val set ----
    print("📊  Evaluating top-1 NN accuracy on validation set …")
    from sklearn.metrics.pairwise import cosine_similarity

    correct = 0
    for _, row in val_df.iterrows():
        q_vec = get_sentence_vector(ft_model, tokenize(row["query_clean"]))
        if np.allclose(q_vec, 0):
            continue
        sims = cosine_similarity(q_vec.reshape(1, -1), train_vectors)[0]
        best_idx = np.argmax(sims)
        predicted = train_df.iloc[best_idx]["category"]
        if predicted == row["category"]:
            correct += 1
    acc = correct / len(val_df) * 100
    print(f"    Val accuracy (FastText, top-1 NN): {acc:.1f}%")

    print("\n✅  All done!  Run the app with:  uvicorn main:app --reload")


if __name__ == "__main__":
    main()
