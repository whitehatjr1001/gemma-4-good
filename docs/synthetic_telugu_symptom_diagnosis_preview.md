# Synthetic Telugu Symptom Diagnosis Preview

Source parquet: data/staged/synthetic_telugu/symptom_diagnosis/train.parquet

Rows: 5

## Columns

```text
output_text
input_text
telugu
romanised_telugu
synthetic_telugu
synthetic_task
synthetic_model
```

## First 5 Rows

### Row 1

**input_text**

```text
I've been having a lot of pain in my neck and back. I've also been having trouble with my balance and coordination. I've been coughing a lot and my limbs feel weak.
```

**output_text**

```text
cervical spondylosis
```

**telugu**

```text
అమ్మా, నెట్టు మరియు వెన్నెముక నొప్పి, సమతుల్యత మరియు సమన్వయం సమస్యలు, తీవ్రమైన కఫం, శరీరం బలహీనత ఉన్నాయి. ఇది తీవ్రమైన లక్షణాలు.
లక్షణాల విశ్లేషణ: నెట్టు నొప్పి, సమతుల్యత సమస్యలు, కఫం, బలహీనత కలిగి ఉంది.
రిస్క్ స్థాయి: HIGH
చర్య: PHC
```

**romanised_telugu**

```text
Amma, nettu mariy and vennemuka noppi, samatulyata mariy samanvaya samasyalu, teevramaina kapham, shariram balahinata unnadi.
Idi teevramaina lakshanaalu.
Lakshanaalu vishleshana: nettu noppi, samatulyata samasyalu, kapham, balahinata kaligi undi.
Risk sthaayi: HIGH
Charya: PHC
```

### Row 2

**input_text**

```text
I have a rash on my face that is getting worse. It is red, inflamed, and has blisters that are bleeding clear pus. It is really painful.
```

**output_text**

```text
impetigo
```

**telugu**

```text
అమ్మా, మీ ముఖంపై ఎర్రటి మచ్చలు, ఎర్రబడిన చర్మం, రక్తపు పాలిపై పుండ్లు ఉన్నాయి. ఇది చాలా నొప్పిగా ఉందా?
లక్షణాల విశ్లేషణ: ఎర్రబడిన చర్మం, పుండ్లు, రక్తపు పాలిపై పుండ్లు, నొప్పి.
రిస్క్ స్థాయి: HIGH
చర్య: PHC
```

**romanised_telugu**

```text
Amma, mi mukhampai errati machalu, errabadi cherman, raktapu paalipai pundlu unnayi. Idi chala noppi gaa undaa?
Lakshanaa vishleshana: Errabadi cherman, pundlu, raktapu paalipai pundlu, noppi.
Risk sthaayi: HIGH
Charya: PHC
```

### Row 3

**input_text**

```text
I have been urinating blood. I sometimes feel sick to my stomach when I urinate. I often feel like I have a fever.
```

**output_text**

```text
urinary tract infection
```

**telugu**

```text
అమ్మా, మూత్రంలో రక్తం వస్తుంది. మూత్రం వేసేటప్పుడు వాంతులు వస్తాయి. ఎండలో ఉంటే జ్వరం అనిపిస్తుంది.
లక్షణాల విశ్లేషణ: మూత్రంలో రక్తం, మూత్రం వేసేటప్పుడు వాంతులు, జ్వరం.
రిస్క్ స్థాయి: HIGH
చర్య: PHC
```

**romanised_telugu**

```text
Amma, mootramlo raktam vastundi. Mootram vesetappudi vaantuluvasthai. Endalo untete jwaram ani pishtundhi.
Lakshanaa vishleshana: Mootramlo raktam, mootram vesetappudi vaantuluvasthai, jwaram.
Risk sthaayi: HIGH
Charya: PHC
```

### Row 4

**input_text**

```text
I have been having trouble with my muscles and joints. My neck is really tight and my muscles feel weak. I have swollen joints and it is hard to move around without becoming stiff. It is also really uncomfortable to walk.
```

**output_text**

```text
arthritis
```

**telugu**

```text
అమ్మా, మీ కండరాలు, జాయింట్స్ నొప్పి ఉన్నాయా? గుండె కింద నెట్టుకుపోయి ఉందా? మీ కండరాలు బలహీనంగా ఉన్నాయా? జాయింట్స్ వీపులు ఉన్నాయా? నడక చాలా కష్టంగా ఉందా?
లక్షణాల విశ్లేషణ: కండరాల బలహీనత, జాయింట్స్ వీపులు, నడక కష్టం. నెట్టుకుపోయిన భావం ఉంది.
రిస్క్ స్థాయి: MEDIUM
చర్య: PHC
```

**romanised_telugu**

```text
Amma, mi kandrala, joints noppi unnaya? Gundhe kinta nettukupo yidha? Mi kandrala balahinanga unnaya? Joints veepulu unnaya? Nadaaka chaala kasthanga unnada?
Lakshanaa vishlesana: Kandrala balahinata, joints veepulu, nadaaka kasthanga. Nettukupo yin bhaavam undi.
Risk sthaayi: MEDIUM
Charya: PHC
```

### Row 5

**input_text**

```text
I have been feeling really sick. My body hurts a lot and I have no appetite. I have also developed rashes on my arms and face. The back of my eyes hurt a lot.
```

**output_text**

```text
dengue
```

**telugu**

```text
అమ్మా, మీ లక్షణాలు ఏమిటి? శరీరం నొప్పి, ఆహారం లేకపోవడం, చేతులు, ముఖంపై రాస్ ఉంది, కళ్ళ వెనుక నొప్పి.
లక్షణాల విశ్లేషణ: రాస్, నొప్పి, ఆహారం లేకపోవడం కలిసి ఉంది. కళ్ళ వెనుక నొప్పి గంభీరంగా ఉంది.
రిస్క్ స్థాయి: HIGH
చర్య: వెంటనే PHCకి వెళ్లండి
```

**romanised_telugu**

```text
Amma, mi lakshanaalu emiti? Shariram noppi, aaharam lekappovadham, chethul, mukhampai ras undi, kalala venuk noppi.
Lakshanaalu vishleshana: Ras, noppi, aaharam lekappovadham kalisi undi. Kalala venuk noppi gambheeraanga undi.
Risk sthaayi: HIGH
Charya: Ventana PHC ki velandi
```
