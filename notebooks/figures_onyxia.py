
# Variance by continent (Heteroskedasticity)
fig, ax = plt.subplots(figsize=(10, 5))


if 'sigma_cluster' in idata.posterior:
    #
    sigma_draws = idata.posterior['sigma_cluster'].values 
    
    for k in range(1, K_clusters+1):
        # le cluster 'k' est à 'k-1'
        draws_k = sigma_draws[..., k-1].flatten()
        ax.violinplot(draws_k, positions=[k], widths=0.6, showmedians=True)

ax.set_xticks(range(1, K_clusters+1))
ax.set_xticklabels([CONTINENT_NAMES.get(k, f'C{k}') for k in range(1, K_clusters+1)])
ax.set_ylabel("σ_cluster (Volatility)")
ax.grid(True, alpha=0.3, axis='y')
plt.title("Geographic Heteroscedasticity (Variance per Cluster)")
plt.tight_layout()
plt.savefig('variance_cluster.pdf', bbox_inches='tight')
plt.show()

# Gravity Coefficients Forest Plot
if 'beta_grav' in idata.posterior:
    beta_flat = idata.posterior['beta_grav'].values.reshape(-1, K_grav)
    beta_means = beta_flat.mean(axis=0)
    beta_q05, beta_q95 = np.percentile(beta_flat, [5, 95], axis=0)

    order = np.argsort(beta_means)
    fig, ax = plt.subplots(figsize=(10, max(6, K_grav * 0.4)))
    colors_coef = ['#F44336' if beta_q05[i] > 0 or beta_q95[i] < 0 else '#90A4AE' for i in order]
    
    ax.barh(range(K_grav), beta_means[order], xerr=[beta_means[order]-beta_q05[order], beta_q95[order]-beta_means[order]], color=colors_coef, alpha=0.8, capsize=3)
    ax.set_yticks(range(K_grav))
    ax.set_yticklabels([X_VOL_COLS[i] for i in order], fontsize=9)
    ax.axvline(0, color='black', lw=1, linestyle='--')
    plt.title("Gravity β Coefficients (90% CI)")
    plt.tight_layout()
    plt.savefig('gravity_coefficients.pdf', bbox_inches='tight')
    plt.show()






# 11. Out-of-Sample Performance Evaluation (2015 test)







y_true_test = df_test['flow'].values
y_pred_test = np.median(fit.stan_variable('flow_test_hat'), axis=0)

# A. Hurdle Accuracy (Binary classification)
y_true_bin = (y_true_test > 0).astype(int)
y_pred_bin = (y_pred_test > 0).astype(int)
acc = accuracy_score(y_true_bin, y_pred_bin)
print(f"OOS Hurdle Accuracy (Binary Open/Close): {acc*100:.1f}%")

# B. Conditional Volume MAE (Only where true flow > 0)
mask_positive = y_true_test > 0
y_true_pos = y_true_test[mask_positive]
y_pred_pos = y_pred_test[mask_positive]
cond_mae = np.mean(np.abs(y_true_pos - y_pred_pos))
print(f"OOS Conditional MAE (True > 0 only)    : {cond_mae:.0f} migrants")

# C. GLOBAL Performance (All dyads, penalizing false positives/negatives)
global_mae = np.mean(np.abs(y_true_test - y_pred_test))
# WMAPE (Weighted Mean Absolute Percentage Error)
global_wmape = np.sum(np.abs(y_true_test - y_pred_test)) / np.sum(y_true_test) * 100

print(f"OOS GLOBAL WMAPE (Weighted Percentage) : {global_wmape:.1f} %")
#  log(1 + x) to safely handle true zeros
global_log_mae = np.mean(np.abs(np.log1p(y_true_test) - np.log1p(y_pred_test)))

print(f"OOS GLOBAL MAE (All dyads)             : {global_mae:.0f} migrants")
print(f"OOS GLOBAL Log-MAE (For ML comparison) : {global_log_mae:.3f}")

# Actual vs Predicted Scatter (Log Scale)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true_test, y_pred_test, alpha=0.5, color='#1565C0', edgecolors='k')
lim = [0, max(y_true_test.max(), y_pred_test.max())]
ax.plot(lim, lim, 'k--', label='Perfect Prediction')
ax.set_xscale('symlog')
ax.set_yscale('symlog')
ax.set_xlabel('True Flow (2015)')
ax.set_ylabel('Predicted Flow (2015)')
ax.legend()
plt.title(f'Global Out-Of-Sample Validation (Log-MAE: {global_log_mae:.3f})')
plt.tight_layout()
plt.savefig('prediction_scatter.pdf', bbox_inches='tight')
plt.show()