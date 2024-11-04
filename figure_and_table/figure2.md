## 2. Sensetivity analysis :

### a. number of data

- use our features on country1 dataset (30k)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/6b1c69fa-400e-40ac-bfa3-aa6df89912bf/image.png)

- use our features on US dataset

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/ae0ec365-b488-4251-b93f-afbc36d58d4a/image.png)

### b. diversity of data source:

- 20% US with different size of country1:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/43faa3b7-6fcc-4f33-b64a-167c087d80b8/image.png)

(20% US + 100% country1: 1/2 US + 1/2 country1)

### c. modality:

- demo_cols = ['age', 'race', 'veteran', 'gender', 'BMI' ]
- textual_cols = note_embedding and entity_present
- clinical_cols = previous_antibiotic_exposure, previous_antibiotic_resistance, specimen type, hospital department

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/1a9387ca-2160-464c-a8e4-099728b9c2f2/image.png)

## 3. train on one dataset, test on another:

- train on country1, test on US

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/791d5bba-8e66-47e1-900b-e4ed03310d1d/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/e42e8cb4-3339-4a76-81b4-64a1499f8ea3/image.png)

nitroflurantoin is generally harder to predict 

- train on US, test on country1

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/6d9d4608-a130-4bba-a871-29c5f74e5054/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/8d012579-e2ae-488e-a415-8bd5f209def5/image.png)

## 4. Oversampling:

conduct oversampling on the whole dataset(US+country1)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/836f602b-0f33-4190-89dd-b06f15312deb/image.png)

## 5. overfitting figure:

overfitting in oversampling case

- LR_AUROC_curve_resistance_ciprofloxacin: unbalanced vs balanced

![LR_AUROC_curve_resistance_ciprofloxacin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/1ff774e9-aa32-4695-af71-3becc778f2a3/LR_AUROC_curve_resistance_ciprofloxacin.png)

![LR_AUROC_curve_resistance_ciprofloxacin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/e916e645-2e45-48dd-8fa6-0f13e7e4d75f/LR_AUROC_curve_resistance_ciprofloxacin.png)

- LR_AUROC_curve_resistance_nitrofurantoin: unbalanced vs balanced

![LR_AUROC_curve_resistance_nitrofurantoin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/a0a0fa4a-ef40-406f-9961-8ac5f0401781/LR_AUROC_curve_resistance_nitrofurantoin.png)

![LR_AUROC_curve_resistance_nitrofurantoin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/0f0b88cd-e11e-441f-a047-ab7653841a5b/LR_AUROC_curve_resistance_nitrofurantoin.png)

- LR_AUROC_curve_resistance_sulfamethoxazole: unbalanced vs balanced

![LR_AUROC_curve_resistance_sulfamethoxazole.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/7aca4be8-b938-45b8-8430-f9d5a256ed7d/LR_AUROC_curve_resistance_sulfamethoxazole.png)

![LR_AUROC_curve_resistance_sulfamethoxazole.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/01a3721b-af2b-4a43-a9f7-437ddc21615f/LR_AUROC_curve_resistance_sulfamethoxazole.png)

- LR_AUROC_curve_resistance_levofloxacin: unbalanced vs balanced

![LR_AUROC_curve_resistance_levofloxacin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/5b10fb28-dc79-4315-bafe-2a03c41b08ee/LR_AUROC_curve_resistance_levofloxacin.png)

![LR_AUROC_curve_resistance_levofloxacin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/266f60bd-e127-4222-b189-9c55f8cc4a2e/LR_AUROC_curve_resistance_levofloxacin.png)

- XBGoost_AUROC_curve_resistance_ciprofloxacin:  unbalanced vs balanced

![XBGoost_AUROC_curve_resistance_ciprofloxacin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/120d31c7-c24d-4e2c-a810-1c1f82bcbbd2/XBGoost_AUROC_curve_resistance_ciprofloxacin.png)

![XBGoost_AUROC_curve_resistance_ciprofloxacin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/d39f1ad9-d8b0-4aee-a82e-d19313b802f9/XBGoost_AUROC_curve_resistance_ciprofloxacin.png)

- XBGoost_AUROC_curve_resistance_levofloxacin:  unbalanced vs balanced

![XBGoost_AUROC_curve_resistance_levofloxacin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/72897670-064d-4a43-8eee-e5062348ed4f/XBGoost_AUROC_curve_resistance_levofloxacin.png)

![XBGoost_AUROC_curve_resistance_levofloxacin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/b2ec6b12-bfdc-4a0d-a5c7-4d3ae61d6148/XBGoost_AUROC_curve_resistance_levofloxacin.png)

- XBGoost_AUROC_curve_resistance_nitrofurantoin:  unbalanced vs balanced

![XBGoost_AUROC_curve_resistance_nitrofurantoin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/6f1eb9c1-c8e4-43fe-8372-23956c055fe5/XBGoost_AUROC_curve_resistance_nitrofurantoin.png)

![XBGoost_AUROC_curve_resistance_nitrofurantoin.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/b2708ef6-3835-44b6-a2cf-4dc431c1af10/XBGoost_AUROC_curve_resistance_nitrofurantoin.png)

- XBGoost_AUROC_curve_resistance_sulfamethoxazole:  unbalanced vs balanced

![XBGoost_AUROC_curve_resistance_sulfamethoxazole.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/b7cd17cf-861c-4041-9c18-d112c221ea2c/XBGoost_AUROC_curve_resistance_sulfamethoxazole.png)

![XBGoost_AUROC_curve_resistance_sulfamethoxazole.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/bff5c4c9-d3a2-4920-8565-8489189f433d/8ed79e2d-0e9f-41e7-b107-9f9d9d7e5b2a/XBGoost_AUROC_curve_resistance_sulfamethoxazole.png)