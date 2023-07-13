

#ifndef LLVM_ANALYSIS_LOADSTOREANALYSIS_H
#define LLVM_ANALYSIS_LOADSTOREANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include <optional>
using namespace llvm;

namespace llvm {

class LoadStoreSourceExpression {
public:
  // Constructor that takes a Function reference.
  LoadStoreSourceExpression(const Function &F) : F(F) {}

  // Print out the values currently in the cache.
  void print(raw_ostream &OS) const;

  // Get the expression string corresponding to an opcode.
  std::string getExpressionFromOpcode(llvm::StringRef opcode);

  // Build the source-level expression for an LLVM instruction.
  void buildSourceLevelExpression(llvm::Instruction &I,
                                                        StringRef symbol);
  
    // This map stores the source-level expressions for LLVM values.
  // The expressions are represented as strings and are associated with the
  // corresponding values. It is used to cache and retrieve source expressions
  // during the generation process.
  llvm::DenseMap<llvm::Value *, std::string> sourceExpressionsMap;

private:
  // This map associates StoreInst pointers with their corresponding LoadInst
  // pointers. It is used to track the relationship between store and load
  // instructions for later processing.
  llvm::DenseMap<llvm::StoreInst *, llvm::LoadInst *> loadStoreMap;

  const Function &F;

  // Remove the ampersand character from a string.
  std::string removeAmpersand(llvm::StringRef str);

  // It is used to store information about the members of a structure.
  llvm::DenseMap<StringRef, std::vector<std::pair<std::string, std::string>>>
      memberInfo;

  // It processes the basePointer to obtain its type information and returns the
  // result as an std::optional<std::string>.
  std::optional<std::string> processBasePointer(llvm::Value *basePointer);

  //  It returns an std::optional<std::string> that may contain a string
  //  representing the processed type information.
  std::optional<std::string> processDIType(llvm::DIType *diType,
                                           llvm::Value *basePointer,
                                           std::string memberName = "");

  //  It is used to track whether a certain array type has been encountered or
  //  not.
  std::unordered_map<std::string, bool> checkArrayType;

  // Get the source-level expression for an LLVM value.
  std::string getSourceExpression(
      llvm::Value *operand,
      llvm::DenseMap<llvm::Value *, std::string> &sourceExpressionsMap,
      llvm::StringRef symbol);

  // Process a StoreInst instruction and return its source-level expression.
  std::string processStoreInst(
      llvm::StoreInst *I,
      llvm::DenseMap<llvm::Value *, std::string> &sourceExpressionsMap,
      llvm::StringRef symbol, bool loadFlag = false);

  // Process a LoadInst instruction and update the sourceExpressionsMap.
  void processLoadInst(
      llvm::LoadInst *I,
      llvm::DenseMap<llvm::Value *, std::string> &sourceExpressionsMap,
      llvm::StringRef symbol);
};

class LoadStoreAnalysis : public AnalysisInfoMixin<LoadStoreAnalysis> {
  friend AnalysisInfoMixin<LoadStoreAnalysis>;
  static AnalysisKey Key;

public:
  using Result = LoadStoreSourceExpression;
  Result run(Function &F, FunctionAnalysisManager &);
//  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class LoadStoreAnalysisPrinterPass
    : public PassInfoMixin<LoadStoreAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit LoadStoreAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif
