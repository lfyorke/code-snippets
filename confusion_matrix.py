# Seaborn and matplotlib confusion matrix

print(classification_report(predictions, y_test))


# In[9]:


mat = confusion_matrix(y_test, predictions)

sns.heatmap(mat.T, square = True, annot = True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')