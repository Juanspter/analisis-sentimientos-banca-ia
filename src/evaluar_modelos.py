for modelo, grupo in df.groupby(col_modelo):
    reales = grupo[col_real]
    predicciones = grupo[col_pred]

    resultados.append({
        "modelo": modelo,
        "accuracy": accuracy_score(reales, predicciones),
        "precision": precision_score(
            reales, predicciones,
            average="macro", zero_division=0
        ),
        "recall": recall_score(
            reales, predicciones,
            average="macro", zero_division=0
        ),
        "f1": f1_score(
            reales, predicciones,
            average="macro", zero_division=0
        )
    })

tabla = pd.DataFrame(resultados)
print(tabla.to_string(index=False))

tabla.to_csv("results/metricas_recalculadas.csv", index=False)
