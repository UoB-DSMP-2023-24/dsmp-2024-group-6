

#install.packages("foreach")
#install.packages("doParallel")
#install.packages("stringdist")
# Load the packages
library(foreach)
library(doParallel)
library(stringdist)

registerDoParallel(cores = detectCores())

file_path <- "C:/Users/james/OneDrive/Documents/Bristol/Mini Project/James Code/turbogliph/gliph-input.txt"

cdr3_sequences <- read.csv(file_path, header = TRUE, sep = "\t")


getRandomSubsample <- function(cdr3_len_stratify = FALSE,
                               vgene_stratify = FALSE,
                               refseqs_motif_region,
                               motif_region,
                               motif_lengths_list,
                               ref_motif_lengths_id_list,
                               motif_region_vgenes_list,
                               ref_motif_vgenes_id_list,
                               ref_lengths_vgenes_list,
                               lengths_vgenes_list){
  random_subsample <- base::c()
  
  ### Return an unbiased reference subsample
  if(cdr3_len_stratify == FALSE && vgene_stratify == FALSE){
    random_subsample <- refseqs_motif_region[base::sample(x = base::seq_along(refseqs_motif_region),size = base::length(motif_region),replace = FALSE)]
  }
  
  ### Return a reference subsample with biased CDR3b length distribution according to the sample
  if(cdr3_len_stratify == TRUE && vgene_stratify == FALSE){
    
    ## if there are more seqs with specific cdr3 length in sample than in reference database, surplus number of seqs will be taken randomly from remaining seqs with different cdr3 length
    random_length <- 0
    for(cdr3_length in base::names(motif_lengths_list)){
      if(base::length(ref_motif_lengths_id_list[[cdr3_length]]) < motif_lengths_list[[cdr3_length]]){
        random_length <- random_length + motif_lengths_list[[cdr3_length]] - base::length(ref_motif_lengths_id_list[[cdr3_length]])
        random_subsample <- base::c(random_subsample, ref_motif_lengths_id_list[[cdr3_length]])
      }else {
        random_subsample <- base::c(random_subsample, base::sample(x = ref_motif_lengths_id_list[[cdr3_length]],size = motif_lengths_list[[cdr3_length]],replace = FALSE))
      }
    }
    if(random_length > 0){
      random_subsample <- base::c(random_subsample, base::sample(x = (base::seq_along(refseqs_motif_region))[-random_subsample],size = random_length,replace = FALSE))
    }
    random_subsample <- refseqs_motif_region[random_subsample]
  }
  
  ### Return a reference subsample with biased V gene distribution according to the sample
  if(vgene_stratify == TRUE && cdr3_len_stratify == FALSE){
    
    ## if there are more seqs with specific v gene in sample than in reference database, surplus number of seqs will be taken randomly from remaining seqs with different vgenes
    random_length <- 0
    for(act_vgene in base::names(motif_region_vgenes_list)){
      if(base::length(ref_motif_vgenes_id_list[[act_vgene]]) < motif_region_vgenes_list[[act_vgene]]){
        random_length <- random_length + motif_region_vgenes_list[[act_vgene]] - base::length(ref_motif_vgenes_id_list[[act_vgene]])
        random_subsample <- base::c(random_subsample, ref_motif_vgenes_id_list[[act_vgene]])
      }else {
        random_subsample <- base::c(random_subsample, base::sample(x = ref_motif_vgenes_id_list[[act_vgene]],size = motif_region_vgenes_list[[act_vgene]],replace = FALSE))
      }
    }
    if(random_length > 0){
      random_subsample <- base::c(random_subsample, base::sample(x = (base::seq_along(refseqs_motif_region))[-random_subsample],size = random_length,replace = FALSE))
    }
    random_subsample <- refseqs_motif_region[random_subsample]
  }
  
  ### Return a reference subsample with biased V gene and CDR3b length distribution according to the sample
  if(vgene_stratify == TRUE && cdr3_len_stratify == TRUE){
    
    ## if there are more seqs with specific v gene or CDR3b length in sample than in reference database, surplus number of seqs will be taken randomly from remaining seqs
    random_length <- 0
    for(cdr3_length in base::names(motif_lengths_list)){
      for(act_vgene in base::names(motif_region_vgenes_list)){
        if(base::length(ref_lengths_vgenes_list[[cdr3_length]][[act_vgene]]) < lengths_vgenes_list[[cdr3_length]][[act_vgene]]){
          random_length <- random_length + lengths_vgenes_list[[cdr3_length]][[act_vgene]] - base::length(ref_lengths_vgenes_list[[cdr3_length]][[act_vgene]])
          random_subsample <- base::c(random_subsample, ref_lengths_vgenes_list[[cdr3_length]][[act_vgene]])
        }else {
          random_subsample <- base::c(random_subsample, base::sample(x = ref_lengths_vgenes_list[[cdr3_length]][[act_vgene]],size = lengths_vgenes_list[[cdr3_length]][[act_vgene]],replace = FALSE))
        }
      }
    }

    if(random_length > 0){
      random_subsample <- base::c(random_subsample, base::sample(x = (base::seq_along(refseqs_motif_region))[-random_subsample],size = random_length,replace = FALSE))
    }
    random_subsample <- refseqs_motif_region[random_subsample]
  }

  ## Closing time!
  base::return(random_subsample)





find_motifs <- function(seqs, q = 2:4,kmer_mindepth = NULL, discontinuous = FALSE){
  all_kmers <- base::c()
  seqs <- base::as.character(seqs)
  all_q <- q
  if(discontinuous == TRUE) all_q <- base::sort(base::unique(base::c(all_q, q+1)))
  
  ### Iterate for all kmer lengths
  for(i in all_q){
    ## Get all continous motifs with length i
    cont_kmer <- stringdist::qgrams(seqs,q = i)
    cont_kmer <- base::as.data.frame(base::t(cont_kmer))
    cont_kmer$motif <- base::rownames(cont_kmer)

    ## Exclude motifs with less counts than kmer_mindepth
    if(!base::is.null(kmer_mindepth)) cont_kmer <- cont_kmer[base::which(cont_kmer$V1 >= kmer_mindepth),] #dplyr::filter(cont_kmer,(V1 >= kmer_mindepth))
    cont_kmer <- cont_kmer[,2:1]
    
    ## Save motifs and corresponding counts
    if(i %in% q) all_kmers <- base::rbind(all_kmers, cont_kmer)

    ## Get all discontinuous motifs with (i-1) fixed positions with i in q
    disc_kmer <- base::c()
    if(discontinuous == TRUE && i %in% (q+1)) {
      ## Find all discontinuous motifs with (i-1) fixed positions and variable position j
      for(j in 2:(i-1)) {
        ## Replace position j by a dot in all continuous motifs  with length i
        disc_kmer <- cont_kmer
        base::substr(disc_kmer$motif, j, j) <- "."
        disc_kmer <- base::as.data.frame(base::table(base::rep(disc_kmer$motif, disc_kmer$V1)))
        base::colnames(disc_kmer) <- base::c("motif", "V1")

        ## Exclude motifs with less counts than kmer_mindepth
        if(!base::is.null(kmer_mindepth)) disc_kmer <- disc_kmer[base::which(disc_kmer$V1 >= kmer_mindepth),] #dplyr::filter(disc_kmer,(V1 >= kmer_mindepth))
        
        ## Save motifs and corresponding counts
        all_kmers <- base::rbind(all_kmers, disc_kmer)
      }
    }
  }

  ## Closing time!
  base::return(all_kmers)
}




cluster_scoring <- function(cluster_list,
                            cdr3_sequences,
                            refdb_beta = "gliph_reference",
                            v_usage_freq = NULL,
                            cdr3_length_freq = NULL,
                            ref_cluster_size = "original",
                            gliph_version = 1,
                            sim_depth = 1000,
                            hla_cutoff = 0.1,
                            n_cores = 1){
  ##################################################################
  ##                         Unit-testing                         ##
  ##################################################################
  
  ### cluster_list
  if(!base::is.list(cluster_list)) base::stop("parameter 'cluster_list' has to be an object of class 'list'.")
  
  ### cdr3_sequences
  if(base::is.atomic(cdr3_sequences)) cdr3_sequences <- base::data.frame("CDR3b" = cdr3_sequences)
  if(!base::is.data.frame(cdr3_sequences)) base::stop("parameter 'cdr3_sequences' has to be an object of class 'data.frame'.")
  cdr3_sequences[] <- base::lapply(cdr3_sequences, as.character)
  
  ### refdb_beta
  # if(!(refdb_beta %in% base::c("gliph_reference", "human_v1.0_CD4", "human_v1.0_CD8", "human_v1.0_CD48", "human_v2.0_CD4",
  #                        "human_v2.0_CD8", "human_v2.0_CD48", "mouse_v1.0_CD4", "mouse_v1.0_CD8", "mouse_v1.0_CD48")) &&
  #    !base::is.data.frame(refdb_beta)){
  if(!base::is.data.frame(refdb_beta)){
    if(base::length(refdb_beta) != 1 || !is.character(refdb_beta)){
      base::stop("refdb_beta has to be a data frame (containing CDR3b sequences in the first column and optional V-gene information in the second column) or the value 'gliph_reference'")
    } else if(!(refdb_beta %in% base::c("gliph_reference"))){
      base::stop("refdb_beta has to be a data frame (containing CDR3b sequences in the first column and optional V-gene information in the second column) or the value 'gliph_reference'")
    }
  }
  
  ### v_usage_freq
  if(!base::is.null(v_usage_freq)){
    if(base::is.data.frame(v_usage_freq)){
      if(base::ncol(v_usage_freq) < 2) base::stop("v_usage_freq has to be a data frame containing V-gene information in the first column and the corresponding frequency in a naive  T-cell repertoire in the second column.")
      if(base::nrow(v_usage_freq) < 1) base::stop("v_usage_freq has to contain at least one row.")
      if(base::suppressWarnings(base::any(base::is.na(base::as.numeric(v_usage_freq[,2])))) == TRUE){
        base::stop("The second column of v_usage_freq must contain the frequency of the corresponding V-gene in the first column in a naive T-cell repertoire.")
      } else v_usage_freq[,2] <- as.numeric(v_usage_freq[,2])
      
    } else {base::stop("v_usage_freq has to be a data frame containing V-gene information in the first column and the corresponding frequency in a naive T-cell repertoire in the second column.")}
  }
  
  ### cdr3_length_freq
  if(!base::is.null(cdr3_length_freq)){
    if(base::is.data.frame(cdr3_length_freq)){
      if(base::ncol(cdr3_length_freq) < 2) base::stop("cdr3_length_freq has to be a data frame containing CDR3 lengths in the first column and the corresponding frequency in a naive  T-cell repertoire in the second column.")
      if(base::nrow(cdr3_length_freq) < 1) base::stop("cdr3_length_freq has to contain at least one row.")
      if(base::suppressWarnings(base::any(base::is.na(base::as.numeric(cdr3_length_freq[,2])))) == TRUE){
        base::stop("The second column of cdr3_length_freq must contain the frequency of the corresponding CDR3 length in the first column in a naive T-cell repertoire.")
      } else cdr3_length_freq[,2] <- as.numeric(cdr3_length_freq[,2])
      
    } else {base::stop("cdr3_length_freq has to be a data frame containing CDR3 lengths in the first column and the corresponding frequency in a naive T-cell repertoire in the second column.")}
  }
  
  ### ref_cluster_size
  if(!(ref_cluster_size %in% base::c("original", "simulated") || !base::is.character(ref_cluster_size) || base::length(ref_cluster_size) > 1)){
    base::stop("ref_cluster_size has to be either 'original' or 'simulated'.")
  }
  
  ### gliph_version
  if(!(gliph_version %in% base::c(1,2))) base::stop("gliph_version has to be either 1 or 2.")
  
  ### sim_depth
  if(!base::is.numeric(sim_depth)) base::stop("sim_depth has to be numeric")
  if(base::length(sim_depth) > 1) base::stop("sim_depth has to be a single number")
  if(sim_depth < 1) base::stop("sim_depth must be at least 1")
  sim_depth <- base::round(sim_depth)
  
  ### hla_cutoff
  if(!base::is.numeric(hla_cutoff)) base::stop("hla_cutoff has to be numeric")
  if(base::length(hla_cutoff) > 1) base::stop("hla_cutoff has to be a single number")
  if(hla_cutoff > 1 || hla_cutoff < 0) base::stop("hla_cutoff must be between 0 and 1")
  
  ### n_cores
  if(base::is.null(n_cores)) n_cores <- base::max(1, parallel::detectCores()-2) else {
    if(!base::is.numeric(n_cores)) base::stop("n_cores has to be numeric")
    if(base::length(n_cores) > 1) base::stop("n_cores has to be a single number")
    if(n_cores < 1) base::stop("n_cores must be at least 1")

    n_cores <- base::min(n_cores, parallel::detectCores()-2)
  }
  
  #################################################################
  ##                         Preparation                         ##
  #################################################################

  ### Which scores can be calculated from the dataset?
  score_names <- base::c("network.size.score","cdr3.length.score")
  if("TRBV" %in% base::colnames(cdr3_sequences)){
    vgene_info <- TRUE
    score_names <- base::c(score_names, "vgene.score")
  } else vgene_info <- FALSE
  if("counts" %in% base::colnames(cdr3_sequences)) {
    counts_info <- TRUE
    cdr3_sequences$counts <- base::as.numeric(cdr3_sequences$counts)
    cdr3_sequences$counts[base::is.na(cdr3_sequences$counts)] <- 1
    score_names <- base::c(score_names, "clonal.expansion.score")
  } else counts_info <- FALSE
  if("patient" %in% base::colnames(cdr3_sequences)) patient_info <- TRUE else patient_info <- FALSE
  if("HLA" %in% base::colnames(cdr3_sequences)) hla_info <- TRUE else hla_info <- FALSE
  if(hla_info == TRUE && patient_info == TRUE){
    if("patient" %in% base::colnames(cdr3_sequences) && "HLA" %in% base::colnames(cdr3_sequences)){
      cdr3_sequences <- cdr3_sequences[cdr3_sequences$HLA != "" & !base::is.na(cdr3_sequences$HLA),]
      if(nrow(cdr3_sequences > 0)) score_names <- base::c(score_names, "hla.score", "lowest.hlas") else hla_info <- FALSE
    }
  }

  ### load or generate reference tables from reference database
  # ref_cluster_sizes:            data frame containing the cluster size in the first and the probability of observing a cluster with this size
  #                               in a sample from the reference database in the second column
  # vgene_ref_frequencies:        vector containing the (relative) frequencies of v gene usage
  # cdr3_length_ref_frequencies:  vector containing the (relative) frequencies of CDR3b lengths
  utils::data(ref_cluster_sizes, envir = base::environment(), package = "turboGliph")
  ref_cluster_sizes <- ref_cluster_sizes[[ref_cluster_size]]

  # maybe changed in future
  reference_list <- NULL
  refdb_name <- "gliph_reference"
  utils::data("reference_list",envir = base::environment(), package = "turboGliph")
  vgene_ref_frequencies <- reference_list[[refdb_name]][[2]]$freq
  cdr3_length_ref_frequencies <- reference_list[[refdb_name]][[3]]$freq
  
  if(!base::identical(refdb_beta, "gliph_reference")){
    if("TRBV" %in% base::colnames(refdb_beta)){
      vgene_ref_frequencies <- base::as.data.frame(base::table(refdb_beta$TRBV))
      vgene_ref_frequencies <- vgene_ref_frequencies$Freq/sum(vgene_ref_frequencies$Freq)
    }
    cdr3_length_ref_frequencies <- base::as.data.frame(base::table(base::nchar(refdb_beta$CDR3b)))
    cdr3_length_ref_frequencies <- cdr3_length_ref_frequencies$Freq/sum(cdr3_length_ref_frequencies$Freq) 
  }
  
  if(!base::is.null(v_usage_freq)) vgene_ref_frequencies <- base::as.numeric(v_usage_freq[,2])
  if(!base::is.null(cdr3_length_freq)) cdr3_length_ref_frequencies <- base::as.numeric(cdr3_length_freq[,2])

  ### Obtain the distribution of all patients and HLA alleles in the sample
  # all_patients:     vector containing all unique patient indices
  # all_hlas:         vector containing all unique HLA alleles
  # all_patient_hlas: list whose elements are named after the patients and contain the patient's HLA alleles in a vector.
  all_patient_hlas <- base::c()
  if(hla_info == TRUE && patient_info == TRUE){
    cdr3_sequences$patient <- gsub(":.*", "",cdr3_sequences$patient)
    all_patients <- base::sort(base::unique(cdr3_sequences$patient))
    all_patients <- all_patients[!base::is.na(all_patients)]
    all_hlas <- base::unlist(base::strsplit(base::unique(cdr3_sequences$HLA), ","))
    all_hlas <- all_hlas[!base::is.na(all_hlas)]
    all_hlas <- base::sort(base::unique(base::gsub(":.*", "", all_hlas, perl = TRUE)))
    num_patients <- base::length(all_patients)
    num_HLAs <- base::length(all_hlas)

    all_patient_hlas <- base::lapply(all_patients, function(x){
      base::sort(base::unique(base::gsub(":.*", "", base::unlist(base::strsplit(cdr3_sequences$HLA[cdr3_sequences$patient == x][1], ",")), perl = TRUE)))
    })
    base::names(all_patient_hlas) <- all_patients

    all_hlas <- base::data.frame(HLA = all_hlas)
    all_hlas$counts <- base::apply(all_hlas, 1, function(x){
      val <- 0
      for(pat in all_patients){
        if(x %in% all_patient_hlas[[pat]]) val <- val+1
      }
      val
    })
  }

  #################################################################
  ##                           Scoring                           ##
  #################################################################
  
  doParallel::registerDoParallel(n_cores)
  
  actCluster <- NULL
  res <- foreach::foreach(actCluster = base::seq_along(cluster_list)) %dopar% {
    
    ### Get sequences and information of current cluster
    act_seq_infos <- cluster_list[[actCluster]]
    num_members <- base::nrow(act_seq_infos) # number of ALL members
    ori_num_members <- base::length(base::unique(act_seq_infos$CDR3b)) # number of all unique CDR3b sequences
    all_scores <- base::c()
    
    ### Get network size score from lookup table
    score_network_size <- 1
    nearest_sample_size <- base::order(base::abs(1-base::as.numeric(base::colnames(ref_cluster_sizes)[-1])/base::nrow(cdr3_sequences)))[1]
    if(ori_num_members > 100){
      score_network_size <-  ref_cluster_sizes[100,nearest_sample_size+1]
    } else {
      score_network_size <-  ref_cluster_sizes[ori_num_members,nearest_sample_size+1]
    }
    all_scores <- base::c(all_scores, score_network_size)
    
    ### Enrichment of CDR3 length (spectratype) within cluster
    score_cdr3_length <- base::c()
    # calculate score of sample (product of all frequencies)
    pick_freqs <- base::data.frame(base::table(base::nchar(base::unique(act_seq_infos$CDR3b))))
    base::colnames(pick_freqs) <- base::c("object", "probs")
    pick_freqs$probs <- pick_freqs$probs/ori_num_members
    sample_score <- base::round(base::prod(pick_freqs$probs), digits = 3)
    
    # generate random subsamples
    random_subsample <- base::list()
    for(i in base::seq_len(sim_depth)){
      random_subsample[[i]] <- base::sample.int(n = base::length(cdr3_length_ref_frequencies), size = ori_num_members,
                                                prob = cdr3_length_ref_frequencies, replace = TRUE)
    }
    
    # calculate score of subsamples (product of all frequencies)
    pick_freqs <- stringdist::seq_qgrams(.list = random_subsample)[,-1]/ori_num_members
    pick_freqs[pick_freqs == 0] <- 1
    pick_freqs <- base::round(base::exp(base::colSums(base::log(pick_freqs))), digits = 3) # vectorized way to calculate the product of each column
    if(gliph_version == 1){
      score_cdr3_length <- base::sum(pick_freqs >= sample_score)/sim_depth
    } else {
      score_cdr3_length <- base::sum(pick_freqs > sample_score)/sim_depth
    }
    if(score_cdr3_length == 0) score_cdr3_length <- 1/sim_depth # minimum score of 1/sim_depth
    
    all_scores <- base::c(all_scores, score_cdr3_length)
    
    ### Enrichment of v genes within cluster
    score_vgene <- base::c()
    if(vgene_info == TRUE){
      
      # calculate score of sample (product of all frequencies)
      pick_freqs <- base::data.frame(base::table(act_seq_infos$TRBV))
      base::colnames(pick_freqs) <- base::c("object", "probs")
      pick_freqs$probs <- pick_freqs$probs/num_members
      sample_score <- base::round(base::prod(pick_freqs$probs), digits = 3)
      
      # generate random subsamples
      random_subsample <- base::list()
      for(i in base::seq_len(sim_depth)){
        random_subsample[[i]] <- base::sample.int(n = base::length(vgene_ref_frequencies), size = num_members,
                                                  prob = vgene_ref_frequencies, replace = TRUE)
      }
      
      # calculate score of subsamples (product of all frequencies)
      pick_freqs <- stringdist::seq_qgrams(.list = random_subsample)[,-1]/num_members
      pick_freqs[pick_freqs == 0] <- 1
      pick_freqs <- base::round(base::exp(base::colSums(base::log(pick_freqs))), digits = 3) # vectorized way to calculate the product of each column
      if(gliph_version == 1){
        score_vgene <- base::sum(pick_freqs >= sample_score)/sim_depth
      } else {
        score_vgene <- base::sum(pick_freqs > sample_score)/sim_depth
      }
      
      if(score_vgene == 0) score_vgene <- 1/sim_depth # minimum score of 1/sim_depth
      all_scores <- base::c(all_scores, score_vgene)
    }
    
    ### Enrichment of clonal expansion within cluster
    score_clonal_expansion <- base::c()
    if(counts_info == TRUE){
      sample_score <- base::sum(base::as.numeric(act_seq_infos$counts))/num_members
      counter <- 0
      for(i in base::seq_len(sim_depth)){
        random_subsample <- base::sample(x = cdr3_sequences$counts, size = num_members, replace = FALSE)
        test_score <- base::sum(base::as.numeric(random_subsample))/num_members
        if(test_score>=sample_score) counter <- counter+1
      }
      if(counter == 0) score_clonal_expansion <- 1/sim_depth else score_clonal_expansion <- counter/sim_depth
      score_clonal_expansion <- base::round(score_clonal_expansion, digits = 3)
      
      all_scores <- base::c(all_scores, score_clonal_expansion)
    }
    
    ### Enrichment of common HLA among donor TCR contributors in cluster
    score_hla <- base::c()
    lowest_hla <- ""
    if(hla_info == TRUE && patient_info == TRUE){
      act_seq_infos <- act_seq_infos[act_seq_infos$HLA != "" & !base::is.na(act_seq_infos$HLA),]
      
      if(base::nrow(act_seq_infos) > 0){
        act_seq_infos$patient <- gsub(":.*", "",act_seq_infos$patient)
      
      score_hla <- 1
      for(act_hla in base::seq_len(num_HLAs)){
        crg_patient_count <- base::length(base::unique(act_seq_infos$patient))
        crg_patient_hla_count <- base::sum(base::unlist(base::lapply(all_patient_hlas[base::unique(act_seq_infos$patient)], function(x){
          if(all_hlas$HLA[act_hla] %in% x) 1 else 0
        })))
        if(crg_patient_hla_count > 1){
          act_Prob <- base::sum(base::choose(all_hlas$counts[act_hla], crg_patient_hla_count:crg_patient_count)*base::choose(num_patients-all_hlas$counts[act_hla], crg_patient_count-crg_patient_hla_count:crg_patient_count)/base::choose(num_patients, crg_patient_count))
          if(act_Prob<score_hla) score_hla <- act_Prob
          if(act_Prob < hla_cutoff){
            if(lowest_hla == ""){
              lowest_hla <- base::paste(all_hlas$HLA[act_hla],
                                        " [(", crg_patient_hla_count, "/", crg_patient_count, ") vs (",
                                        all_hlas$counts[act_hla], "/", num_patients,
                                        ") = ",
                                        base::formatC(act_Prob, digits = 1, format = "e"),
                                        "]",
                                        sep = "" )
            } else {
              lowest_hla <- base::paste(lowest_hla, ", ", all_hlas$HLA[act_hla],
                                        " [(", crg_patient_hla_count, "/", crg_patient_count, ") vs (",
                                        all_hlas$counts[act_hla], "/", num_patients,
                                        ") = ",
                                        base::formatC(act_Prob, digits = 1, format = "e"),
                                        "]",
                                        sep = "" )
            }
          }
        }
        
      }
      } else {
      score_hla <- 1
      }
      
      all_scores <- base::c(all_scores, score_hla)
    }
    
    ### Total score
    if(gliph_version == 1) score_final <- base::prod(all_scores)*0.001*64 else if(gliph_version == 2) score_final <- base::prod(all_scores)
    
    ### Output
    all_scores <- c(score_final, all_scores)
    all_scores <- base::formatC(all_scores, digits = 1, format = "e")
    output <- base::c(base::names(cluster_list)[actCluster], all_scores)
    if(hla_info == TRUE && patient_info == TRUE){
      output <- base::c(output, lowest_hla)
    }
    output
  }
  
  doParallel::stopImplicitCluster()
  
  res <- base::data.frame(base::matrix(base::unlist(res), ncol = 2+base::length(score_names), byrow = TRUE))
  base::colnames(res) <- base::c("leader.tag", "total.score", score_names)
  
  for(i in base::c("total.score", score_names)) if(i != "lowest.hlas") res[,i] <- base::as.numeric(res[,i])

  # set all theoretical numeric values (actually characters) to numeric values
  if(base::is.data.frame(res)){
    for(i in base::seq_len(base::ncol(res))){
      if(base::suppressWarnings(base::any(base::is.na(base::as.numeric(res[,i])))) == FALSE) res[,i] <- base::as.numeric(res[,i])
    }
  }
  
  # Closing time!
  base::return(res[,-1])
}



getRandomSubsample <- function(cdr3_len_stratify = FALSE,
                               vgene_stratify = FALSE,
                               refseqs_motif_region,
                               motif_region,
                               motif_lengths_list,
                               ref_motif_lengths_id_list,
                               motif_region_vgenes_list,
                               ref_motif_vgenes_id_list,
                               ref_lengths_vgenes_list,
                               lengths_vgenes_list){
  random_subsample <- base::c()
  
  ### Return an unbiased reference subsample
  if(cdr3_len_stratify == FALSE && vgene_stratify == FALSE){
    random_subsample <- refseqs_motif_region[base::sample(x = base::seq_along(refseqs_motif_region),size = base::length(motif_region),replace = FALSE)]
  }
  
  ### Return a reference subsample with biased CDR3b length distribution according to the sample
  if(cdr3_len_stratify == TRUE && vgene_stratify == FALSE){
    
    ## if there are more seqs with specific cdr3 length in sample than in reference database, surplus number of seqs will be taken randomly from remaining seqs with different cdr3 length
    random_length <- 0
    for(cdr3_length in base::names(motif_lengths_list)){
      if(base::length(ref_motif_lengths_id_list[[cdr3_length]]) < motif_lengths_list[[cdr3_length]]){
        random_length <- random_length + motif_lengths_list[[cdr3_length]] - base::length(ref_motif_lengths_id_list[[cdr3_length]])
        random_subsample <- base::c(random_subsample, ref_motif_lengths_id_list[[cdr3_length]])
      }else {
        random_subsample <- base::c(random_subsample, base::sample(x = ref_motif_lengths_id_list[[cdr3_length]],size = motif_lengths_list[[cdr3_length]],replace = FALSE))
      }
    }
    if(random_length > 0){
      random_subsample <- base::c(random_subsample, base::sample(x = (base::seq_along(refseqs_motif_region))[-random_subsample],size = random_length,replace = FALSE))
    }
    random_subsample <- refseqs_motif_region[random_subsample]
  }
  
  ### Return a reference subsample with biased V gene distribution according to the sample
  if(vgene_stratify == TRUE && cdr3_len_stratify == FALSE){
    
    ## if there are more seqs with specific v gene in sample than in reference database, surplus number of seqs will be taken randomly from remaining seqs with different vgenes
    random_length <- 0
    for(act_vgene in base::names(motif_region_vgenes_list)){
      if(base::length(ref_motif_vgenes_id_list[[act_vgene]]) < motif_region_vgenes_list[[act_vgene]]){
        random_length <- random_length + motif_region_vgenes_list[[act_vgene]] - base::length(ref_motif_vgenes_id_list[[act_vgene]])
        random_subsample <- base::c(random_subsample, ref_motif_vgenes_id_list[[act_vgene]])
      }else {
        random_subsample <- base::c(random_subsample, base::sample(x = ref_motif_vgenes_id_list[[act_vgene]],size = motif_region_vgenes_list[[act_vgene]],replace = FALSE))
      }
    }
    if(random_length > 0){
      random_subsample <- base::c(random_subsample, base::sample(x = (base::seq_along(refseqs_motif_region))[-random_subsample],size = random_length,replace = FALSE))
    }
    random_subsample <- refseqs_motif_region[random_subsample]
  }
  
  ### Return a reference subsample with biased V gene and CDR3b length distribution according to the sample
  if(vgene_stratify == TRUE && cdr3_len_stratify == TRUE){
    
    ## if there are more seqs with specific v gene or CDR3b length in sample than in reference database, surplus number of seqs will be taken randomly from remaining seqs
    random_length <- 0
    for(cdr3_length in base::names(motif_lengths_list)){
      for(act_vgene in base::names(motif_region_vgenes_list)){
        if(base::length(ref_lengths_vgenes_list[[cdr3_length]][[act_vgene]]) < lengths_vgenes_list[[cdr3_length]][[act_vgene]]){
          random_length <- random_length + lengths_vgenes_list[[cdr3_length]][[act_vgene]] - base::length(ref_lengths_vgenes_list[[cdr3_length]][[act_vgene]])
          random_subsample <- base::c(random_subsample, ref_lengths_vgenes_list[[cdr3_length]][[act_vgene]])
        }else {
          random_subsample <- base::c(random_subsample, base::sample(x = ref_lengths_vgenes_list[[cdr3_length]][[act_vgene]],size = lengths_vgenes_list[[cdr3_length]][[act_vgene]],replace = FALSE))
        }
      }
    }

    if(random_length > 0){
      random_subsample <- base::c(random_subsample, base::sample(x = (base::seq_along(refseqs_motif_region))[-random_subsample],size = random_length,replace = FALSE))
    }
    random_subsample <- refseqs_motif_region[random_subsample]
  }

  ## Closing time!
  base::return(random_subsample)
}




turbo_gliph <- function(cdr3_sequences,
                        result_folder = "",
                        refdb_beta = "gliph_reference",
                        v_usage_freq = NULL,
                        cdr3_length_freq = NULL,
                        ref_cluster_size = "original",
                        sim_depth = 1000,
                        lcminp = 0.01,
                        lcminove = c(1000, 100, 10),
                        kmer_mindepth = 3,
                        accept_sequences_with_C_F_start_end = TRUE,
                        min_seq_length = 8,
                        gccutoff = NULL,
                        structboundaries = TRUE,
                        boundary_size = 3,
                        motif_length = base::c(2,3,4),
                        discontinuous = FALSE, #Draius30012020: discontinuous motifs added
                        make_depth_fig = FALSE, #Draius
                        local_similarities = TRUE, #Draius13042020
                        global_similarities = TRUE, #Draius13042020,
                        global_vgene = FALSE, #Draius14042020
                        positional_motifs = FALSE, #Draius15042020
                        cdr3_len_stratify = FALSE, #Draius17042020
                        vgene_stratify = FALSE, #Draius17042020
                        public_tcrs = TRUE, #Draius16042020
                        cluster_min_size = 2,
                        hla_cutoff = 0.1,
                        n_cores = 1){
  ##################################################################
  ##                         Unit-testing                         ##
  ##################################################################
  
  ### result_folder
  if(!base::is.character(result_folder)) base::stop("result_folder has to be a character object")
  if(base::length(result_folder) > 1) base::stop("result_folder has to be a single path")
  save_results <- FALSE
  if(result_folder != ""){
    if(base::substr(result_folder,base::nchar(result_folder),base::nchar(result_folder)) != "/") result_folder <- base::paste0(result_folder,"/") # nolint
    if(!base::dir.exists(result_folder)) base::dir.create(result_folder)
    save_results <- TRUE
    if(base::file.exists(base::paste0(result_folder,"kmer_resample_",sim_depth,"_log.txt"))){
      save_results <- FALSE
      base::cat(base::paste("WARNING: Saving the results is cancelled. The following file already exists in the result folder:\n",
                            base::paste0(result_folder,"kmer_resample_",sim_depth,"_log.txt"),"\n"))
    }
    if(base::file.exists(base::paste0(result_folder,"kmer_resample_",sim_depth,"_minp",lcminp,"_ove",base::paste(lcminove, collapse = "_"),".txt"))){
      save_results <- FALSE
      base::cat(base::paste("WARNING: Saving the results is cancelled. The following file already exists in the result folder:\n",
                            base::paste0(result_folder,"kmer_resample_",sim_depth,"_minp",lcminp,"_ove",base::paste(lcminove, collapse = "_"),".txt"),"\n"))
    }
    if(base::file.exists(base::paste0(result_folder,"kmer_resample_",sim_depth,"_all_motifs.txt"))){
      save_results <- FALSE
      base::cat(base::paste("WARNING: Saving the results is cancelled. The following file already exists in the result folder:\n",
                            base::paste0(result_folder,"kmer_resample_",sim_depth,"_all_motifs.txt"),"\n"))
    }
    if(base::file.exists(base::paste0(result_folder, "clone_network.txt"))){
      save_results <- FALSE
      base::cat(base::paste("WARNING: Saving the results is cancelled. The following file already exists in the result folder:\n",
                            base::paste0(result_folder, "clone_network.txt"),"\n"))
    }
    if(base::file.exists(base::paste0(result_folder,"convergence_groups.txt"))){
      save_results <- FALSE
      base::cat(base::paste("WARNING: Saving the results is cancelled. The following file already exists in the result folder:\n",
                            base::paste0(result_folder,"convergence_groups.txt"),"\n"))
    }
    if(base::file.exists(base::paste0(result_folder,"parameter.txt"))){
      save_results <- FALSE
      base::cat(base::paste("WARNING: Saving the results is cancelled. The following file already exists in the result folder:\n",
                            base::paste0(result_folder,"parameter.txt"),"\n"))
    }
  }
  
  ### refdb_beta
  # if(!(refdb_beta %in% base::c("gliph_reference", "human_v1.0_CD4", "human_v1.0_CD8", "human_v1.0_CD48", "human_v2.0_CD4",
  #                        "human_v2.0_CD8", "human_v2.0_CD48", "mouse_v1.0_CD4", "mouse_v1.0_CD8", "mouse_v1.0_CD48")) &&
  #    !base::is.data.frame(refdb_beta)){
  if(!base::is.data.frame(refdb_beta)){
    if(base::length(refdb_beta) != 1 || !is.character(refdb_beta)){
      base::stop("refdb_beta has to be a data frame (containing CDR3b sequences in the first column and optional V-gene information in the second column) or the value 'gliph_reference'")
    } else if(!(refdb_beta %in% base::c("gliph_reference"))){
      base::stop("refdb_beta has to be a data frame (containing CDR3b sequences in the first column and optional V-gene information in the second column) or the value 'gliph_reference'")
    }
  }
  
  ### v_usage_freq
  if(!base::is.null(v_usage_freq)){
    if(base::is.data.frame(v_usage_freq)){
      if(base::ncol(v_usage_freq) < 2) base::stop("v_usage_freq has to be a data frame containing V-gene information in the first column and the corresponding frequency in a naive  T-cell repertoire in the second column.")
      if(base::nrow(v_usage_freq) < 1) base::stop("v_usage_freq has to contain at least one row.")
      if(base::suppressWarnings(base::any(base::is.na(base::as.numeric(v_usage_freq[,2])))) == TRUE){
        base::stop("The second column of v_usage_freq must contain the frequency of the corresponding V-gene in the first column in a naive T-cell repertoire.")
      } else v_usage_freq[,2] <- as.numeric(v_usage_freq[,2])
      
    } else {base::stop("v_usage_freq has to be a data frame containing V-gene information in the first column and the corresponding frequency in a naive T-cell repertoire in the second column.")}
  }
  
  ### cdr3_length_freq
  if(!base::is.null(cdr3_length_freq)){
    if(base::is.data.frame(cdr3_length_freq)){
      if(base::ncol(cdr3_length_freq) < 2) base::stop("cdr3_length_freq has to be a data frame containing CDR3 lengths in the first column and the corresponding frequency in a naive  T-cell repertoire in the second column.")
      if(base::nrow(cdr3_length_freq) < 1) base::stop("cdr3_length_freq has to contain at least one row.")
      if(base::suppressWarnings(base::any(base::is.na(base::as.numeric(cdr3_length_freq[,2])))) == TRUE){
        base::stop("The second column of cdr3_length_freq must contain the frequency of the corresponding CDR3 length in the first column in a naive T-cell repertoire.")
      } else cdr3_length_freq[,2] <- as.numeric(cdr3_length_freq[,2])
      
    } else {base::stop("cdr3_length_freq has to be a data frame containing CDR3 lengths in the first column and the corresponding frequency in a naive T-cell repertoire in the second column.")}
  }
  
  ### ref_cluster_size
  if(!(ref_cluster_size %in% base::c("original", "simulated") || !base::is.character(ref_cluster_size) || base::length(ref_cluster_size) > 1)){
    base::stop("ref_cluster_size has to be either 'original' or 'simulated'.")
  }
  
  ### sim_depth
  if(!base::is.numeric(sim_depth)) base::stop("sim_depth has to be numeric")
  if(base::length(sim_depth) > 1) base::stop("sim_depth has to be a single number")
  if(sim_depth < 1) base::stop("sim_depth must be at least 1")
  sim_depth <- base::round(sim_depth)
  
  ### lcminp
  if(!base::is.numeric(lcminp)) base::stop("lcminp has to be numeric")
  if(base::length(lcminp) > 1) base::stop("lcminp has to be a single number")
  if(lcminp <= 0) base::stop("lcminp must be greater than 0")
  
  ### lcminove
  if(!base::is.numeric(lcminove)) base::stop("lcminove has to be numeric")
  if(base::length(lcminove) > 1 && base::length(lcminove) != base::length(motif_length)) base::stop("lcminove has to be a single number or of same length as motif_length")
  if(base::any(lcminove < 1)) base::stop("lcminove must be at least 1")
  
  ### kmer_mindepth
  if(!base::is.numeric(kmer_mindepth)) base::stop("kmer_mindepth has to be numeric")
  if(base::length(kmer_mindepth) > 1) base::stop("kmer_mindepth has to be a single number")
  if(kmer_mindepth < 1) base::stop("kmer_mindepth must be at least 1")
  kmer_mindepth <- base::round(kmer_mindepth)
  
  ### accept_sequences_with_C_F_start_end
  if(!base::is.logical(accept_sequences_with_C_F_start_end)) base::stop("accept_sequences_with_C_F_start_end has to be logical")
  
  ### min_seq_length
  if(!base::is.numeric(min_seq_length)) base::stop("min_seq_length has to be numeric")
  if(base::length(min_seq_length) > 1) base::stop("min_seq_length has to be a single number")
  if(min_seq_length < 0) base::stop("min_seq_length must be at least 0")
  min_seq_length <- base::round(min_seq_length)
  
  ### gccutoff
  if(!base::is.null(gccutoff) && !base::is.numeric(gccutoff)) base::stop("gccutoff has to be NULL or numeric")
  if(!base::is.null(gccutoff) && base::length(gccutoff)>1) base::stop("gccutoff has to be NULL or a single number")
  if(!base::is.null(gccutoff) && gccutoff < 0) base::stop("gccutoff must be at least 0")
  
  ### structboundaries
  if(!base::is.logical(structboundaries)) base::stop("structboundaries has to be logical")
  
  ### boundary_size
  if(!base::is.numeric(boundary_size)) base::stop("boundary_size has to be numeric")
  if(base::length(boundary_size) > 1) base::stop("boundary_size has to be a single number")
  if(boundary_size < 0) base::stop("boundary_size must be at least 0")
  boundary_size <- base::round(boundary_size)
  if(structboundaries == TRUE) min_seq_length <- base::max(min_seq_length, ((boundary_size * 2)+1))
  
  ### motif_length
  if(!base::is.numeric(motif_length)) base::stop("motif_length has to be numeric")
  if(base::any(motif_length < 1)) base::stop("values of motif_length must be at least 1")
  motif_length <- base::round(motif_length)
  
  ### discontinuous
  if(!base::is.logical(discontinuous)) base::stop("discontinuous has to be logical")
  
  ### make_depth_fig
  if(!base::is.logical(make_depth_fig)) base::stop("make_depth_fig has to be logical")
  
  ### local_similarities
  if(!base::is.logical(local_similarities)) base::stop("local_similarities has to be logical")
  
  ### global_similarities
  if(!base::is.logical(global_similarities)) base::stop("global_similarities has to be logical")
  if(local_similarities == FALSE && global_similarities == FALSE) base::stop("Either local_similarities or global_similarities have to be TRUE")
  
  ### global_vgene
  if(!base::is.logical(global_vgene)) base::stop("global_vgene has to be logical")
  
  ### positional_motifs
  if(!base::is.logical(positional_motifs)) base::stop("positional_motifs has to be logical")
  
  ### cdr3_len_stratify
  if(!base::is.logical(cdr3_len_stratify)) base::stop("cdr3_len_stratify has to be logical")
  
  ### vgene_stratify
  if(!base::is.logical(vgene_stratify)) base::stop("vgene_stratify has to be logical")
  
  ### public_tcrs
  if(!base::is.logical(public_tcrs)) base::stop("public_tcrs has to be logical")
  
  ### cluster_min_size
  if(!base::is.numeric(cluster_min_size)) base::stop("cluster_min_size has to be numeric")
  if(base::length(cluster_min_size) > 1) base::stop("cluster_min_size has to be a single number")
  if(cluster_min_size < 1) base::stop("cluster_min_size must be at least 1")
  cluster_min_size <- base::round(cluster_min_size)
  
  ### hla_cutoff
  if(!base::is.numeric(hla_cutoff)) base::stop("hla_cutoff has to be numeric")
  if(base::length(hla_cutoff) > 1) base::stop("hla_cutoff has to be a single number")
  if(hla_cutoff > 1 || hla_cutoff < 0) base::stop("hla_cutoff must be between 0 and 1")
  
  ### n_cores
  if(!base::is.null(n_cores))
  {
    if(!base::is.numeric(n_cores)) base::stop("n_cores has to be numeric")
    if(base::length(n_cores) > 1) base::stop("n_cores has to be a single number")
    if(n_cores < 1) base::stop("n_cores must be at least 1")
    n_cores <- base::round(n_cores)
  }
  
  ### min_seq_length second round
  if(structboundaries == TRUE) min_seq_length <- base::max(min_seq_length, 2*boundary_size)
  
  t1 <- base::Sys.time()
  
  #################################################################
  ##                      Input preparation                      ##
  #################################################################
  vgene.info <- FALSE
  patient.info <- FALSE
  hla.info <- FALSE
  count.info <- FALSE
  
  ### Check cdr3_sequences for desired structure and information and convert it into a data frame
  if(base::is.atomic(cdr3_sequences)) {
    sequences <- base::data.frame(CDR3b = cdr3_sequences, stringsAsFactors = FALSE)
    if(global_vgene == TRUE){
      base::stop("To restrict global similarities to v-gene sharing, v-gene information is required as column of cdr3_sequences named \"TRBV\"")
    }
    if(vgene_stratify == TRUE) base::stop("For v-gene stratification v-gene information is required as column of cdr3_sequences named \"TRBV\"")
    if(public_tcrs == FALSE) public_tcrs <- TRUE # No donor information, no differentiation
  } else if (base::is.data.frame(cdr3_sequences)) {
    if(base::ncol(cdr3_sequences) == 1){
      sequences <- base::data.frame(CDR3b = cdr3_sequences[,1],
                                    stringsAsFactors = FALSE)
      base::cat("Notification: cdr3_sequences is a data frame and the first column is considered as cdr3 sequences.\n")
    } else {
      if(!("CDR3b" %in% base::colnames(cdr3_sequences))) base::stop("cdr3_sequences has to contain a column named \"CDR3b\" comprising cdr3 beta sequences")
      sequences <- base::data.frame(CDR3b = cdr3_sequences[,1],
                                    stringsAsFactors = FALSE)
      base::cat("Notification: cdr3_sequences is a data frame and the column named \"CDR3b\" is considered as cdr3 beta sequences.\n")
    }
    if("TRBV" %in% base::colnames(cdr3_sequences)){
      vgene.info <- TRUE
      sequences$TRBV <- cdr3_sequences$TRBV
    } else if(global_vgene == TRUE){
      base::stop("To restrict global similarities to v-gene sharing, v-gene information is required as column of cdr3_sequences named \"TRBV\"")
    } else if(vgene_stratify == TRUE){
      base::stop("For v-gene stratification v-gene information is required as column of cdr3_sequences named \"TRBV\"")
    }
    
    if("patient" %in% base::colnames(cdr3_sequences)){
      patient.info <- TRUE
      sequences$patient <- cdr3_sequences$patient
    }
    if("HLA" %in% base::colnames(cdr3_sequences)){
      hla.info <- TRUE
      sequences$HLA <- cdr3_sequences$HLA
    }
    if("counts" %in% base::colnames(cdr3_sequences)){
      count.info <- TRUE
      cdr3_sequences$counts <- base::as.numeric(cdr3_sequences$counts)
      cdr3_sequences$counts[base::is.na(cdr3_sequences$counts)] <- 1
      sequences$counts <- cdr3_sequences$counts
    }
    
    if(base::any(!(base::colnames(cdr3_sequences) %in% base::colnames(sequences)))) sequences <- base::cbind(sequences,
                                                                                                             cdr3_sequences[,!(base::colnames(cdr3_sequences) %in% base::colnames(sequences))])
    
    sequences[] <- base::lapply(sequences, base::as.character)
  } else {
    base::stop("cdr3_sequences must be a character vector or a data frame with sequences in the first column.\n")
  }
  if(patient.info == FALSE && public_tcrs == FALSE){
    base::stop("To restrict clusters to tcrs obtained from one patient, donor information is required as column of cdr3_sequences named \"patient\"")
  }
  if(base::ncol(sequences) == 1){
    sequences <- base::data.frame(CDR3b = sequences[base::grep("^[ACDEFGHIKLMNOPQRSTUVWY]*$", sequences$CDR3b),])
    if(base::nrow(sequences) == 0) base::stop("No CDR3b amino acid sequences found. Check if only letters of the one-letter code are used.")
  } else {
    sequences <- sequences[base::grep("^[ACDEFGHIKLMNPQRSTVWY]*$", sequences$CDR3b),]
    if(base::nrow(sequences) == 0) base::stop("No CDR3b amino acid sequences found. Check if only letters of the one-letter code are used.")
  }
  sequences <- base::cbind(base::data.frame(seq_ID = base::seq_len(base::nrow(sequences))), sequences)
  
  #################################################################
  ##              Preparation of reference database              ##
  #################################################################
  
  ### load non-redundant reference repertoire 
  if(base::is.data.frame(refdb_beta)) {
    refseqs <- refdb_beta
    refseqs[] <- base::lapply(refseqs, base::as.character)
    
    if(base::nrow(refseqs) <= 0) base::stop("User-created reference repertoire is of length 0. Without reference sequences the algorithm is not able to work.")
    
    if("CDR3b" %in% base::colnames(refseqs)){
      base::cat("Notification: Column of reference database named 'CDR3b' is considered as cdr3 sequences.\n")
    } else if(base::ncol(refseqs) > 1){
      base::cat("Notification: First column of reference database is considered as cdr3 sequences.\n")
      base::colnames(refseqs)[1] <- "CDR3b"
    } else {
      base::colnames(refseqs)[1] <- "CDR3b"
    }
    if(base::ncol(refseqs) > 1 && global_vgene == TRUE){
      if("TRBV" %in% base::colnames(refseqs)){
        base::cat("Notification: Column of reference database named 'TRBV' is considered as V gene information.\n")
      } else {
        base::cat("Notification: Second column of reference database is considered as V gene information.\n")
        base::colnames(refseqs)[2] <- "TRBV"
      }
    } else if(global_vgene == TRUE){
      base::stop("V-gene restriction for global similarities ('global_vgene' == TRUE) requires V-gene information in second column of reference database.\n")
    } else {
      refseqs$TRBV <- base::rep("", base::nrow(refseqs))
    }
    if(base::ncol(refseqs) == 1){
      refseqs <- base::cbind(refseqs, base::rep("", base::nrow(refseqs)))
      base::colnames(refseqs) <- base::c("CDR3b", "TRBV")
    }
    
    refseqs <- refseqs[, base::c("CDR3b", "TRBV")]
    
    if(accept_sequences_with_C_F_start_end) refseqs <- refseqs[base::grep(pattern = "^C.*F$",x = refseqs[,1],perl = TRUE),]
    if(base::nrow(refseqs) == 0) base::stop("No reference CDR3b sequences starting with C and ending with F found. Adjust the parameter 'accept_sequences_with_C_F_start_end' for further procedure.")
    refseqs <- base::unique(refseqs)
    refseqs <- refseqs[base::which(base::nchar(refseqs[,1]) >= min_seq_length),]
    if(base::nrow(refseqs) == 0) base::stop("No reference CDR3b sequences with a minimum length of 'min_seq_length' found. Adjust this parameter for further procedure.")
    
    refseqs <- refseqs[base::grep("^[ACDEFGHIKLMNPQRSTVWY]*$", refseqs$CDR3b),]
    if(base::nrow(refseqs) == 0) base::stop("No reference CDR3b amino acid sequences found. Check if only letters of the one-letter code are used.")
    
    if(vgene_stratify == TRUE){
      ref_vgenes <- base::as.character(refseqs$TRBV)
    }
    
    refseqs <- base::as.character(refseqs$CDR3b)
  } else {
    reference_list <- NULL
    utils::data("reference_list",envir = base::environment(), package = "turboGliph")
    refseqs <- base::as.data.frame(reference_list[[refdb_beta]]$refseqs)
    refseqs[] <- base::lapply(refseqs, base::as.character)
    
    if(accept_sequences_with_C_F_start_end) refseqs <- refseqs[base::grep(pattern = "^C.*F$",x = refseqs$CDR3b,perl = TRUE),]
    refseqs <- base::unique(refseqs)
    refseqs <- refseqs[base::which(base::nchar(refseqs$CDR3b) >= min_seq_length),]
    refseqs <- refseqs[base::grep("^[ACDEFGHIKLMNPQRSTVWY]*$", refseqs$CDR3b),]
    
    if(vgene_stratify == TRUE){
      ref_vgenes <- base::as.character(refseqs$TRBV)
      base::cat("Notification: Second column of reference database is considered as v-gene information.\n")
    }
    
    refseqs <- base::as.character(refseqs$CDR3b)
  }
  
  ### Initiate parallel execution
  if(!base::is.null(n_cores)){
    no_cores <- n_cores
  }else{
    if(parallel::detectCores() > 2) {no_cores <- parallel::detectCores() - 2}else{no_cores <- 1}
  }
  doParallel::registerDoParallel(no_cores)
  
  ##################################################################
  ##                  Part 1: Identify all motifs                 ##
  ##################################################################
  if(local_similarities == TRUE){
    base::cat(base::paste("Part 1: Searching for motifs in the data and reference DB. \n"))
    
    
    part_1_res_file_name <- base::paste0(result_folder,"kmer_resample_",sim_depth,"_log.txt")
    
    ### Reduce non-redundant sample and reference sequences to the motif region
    seqs <- sequences$CDR3b
    if(accept_sequences_with_C_F_start_end) seqs <- base::grep(pattern = "^C.*F$",x = sequences$CDR3b,perl = TRUE,value = TRUE)
    if(base::length(seqs) == 0) base::stop("No CDR3b sequences starting with C and ending with F found. Adjust the parameter 'accept_sequences_with_C_F_start_end' for further procedure.")
    seqs <- base::unique(seqs)
    seqs <- seqs[base::which(base::nchar(seqs) >= min_seq_length)]
    if(base::length(seqs) == 0) base::stop("No CDR3b sequences with a minimum length of 'min_seq_length' found. Adjust this parameter for further procedure.")
    if(structboundaries) motif_region <- base::substr(x = seqs,start = boundary_size + 1 ,stop = base::nchar(seqs) - boundary_size) else motif_region <- seqs
    
    if(vgene_stratify == TRUE) full_refseqs <- refseqs
    if(structboundaries) refseqs_motif_region <- base::substr(x = refseqs,start = boundary_size + 1,stop = base::nchar(refseqs)-boundary_size) else refseqs_motif_region <- refseqs
    
    if(base::length(refseqs_motif_region) < base::length(motif_region)) {
      base::stop("Analysis failed. Reference database must have more sequences than input data for the significance analysis to make sense and work.\n")
    }
    
    ### Prepare biased repeated random sampling
    # get v gene distribution for biased repeat random sampling
    motif_region_vgenes_list <- base::list()
    ref_motif_vgenes_id_list <- base::list()
    if(vgene_stratify == TRUE){
      for(act_vgene in base::sort(base::as.character(base::unique(sequences$TRBV)))){
        motif_region_vgenes_ids[[act_vgene]] <- base::sum(sequences$TRBV == act_vgene)
        ref_motif_vgenes_id_list[[act_vgene]] <- base::which(refseqs %in% full_refseqs[ref_vgenes == act_vgene])
      }
    }
    # get CDR3b length distribution for biased repeat random sampling
    motif_lengths_list <- base::list()
    ref_motif_lengths_id_list <- base::list()
    if(cdr3_len_stratify == TRUE){
      motif_region_lengths <- base::nchar(motif_region)
      ref_motif_region_lengths <- base::nchar(refseqs_motif_region)
      
      for(cdr3_length in base::sort(base::unique(motif_region_lengths))){
        motif_lengths_list[[base::as.character(cdr3_length)]] <- base::sum(motif_region_lengths == cdr3_length)
        ref_motif_lengths_id_list[[base::as.character(cdr3_length)]] <- base::which(ref_motif_region_lengths == cdr3_length)
      }
    }
    # get correlated v gene and CDR3b length distribution for biased repeat random sampling
    lengths_vgenes_list <- base::list()
    ref_lengths_vgenes_list <- base::list()
    if((vgene_stratify == TRUE) && (cdr3_len_stratify == TRUE)){
      for(cdr3_length in base::names(ref_motif_lengths_id_list)){
        for(act_vgene in base::names(ref_motif_vgenes_id_list)){
          lengths_vgenes_list[[cdr3_length]][[act_vgene]] <- base::sum(sequences$TRBV == act_vgene & motif_region_lengths == base::as.numeric(cdr3_length))
          ref_lengths_vgenes_list[[cdr3_length]][[act_vgene]] <- ref_motif_lengths_id_list[[cdr3_length]][ref_motif_lengths_id_list[[cdr3_length]] %in% ref_motif_vgenes_id_list[[act_vgene]]]
        }
      }
    }
    
    ### Visualize global convergence by (biased) repeat random sampling from reference database at sample set depth
    if(make_depth_fig == TRUE){
      base::cat("Simulated stochastic resampling of", base::length(motif_region) , "sequences from reference databse.\n")
      counts <- base::rep(0,20)
      # Get minimum distance to nearest neighbor for every sequence in sample set
      for(i in base::seq_along(motif_region)){
        distance <- base::nchar(motif_region[i])
        dis <- base::min(stringdist::stringdist(a = motif_region[i],b = motif_region[-i],method = "hamming",nthread = 1))
        if(dis < distance) distance <- dis
        
        counts[distance] <- counts[distance] + 1
      }
      
      graphics::barplot(counts, names.arg = base::seq_len(20), main = base::paste("Global convergence distribution - ", "Sample", sep = ""), xlab = "Hamming distance", ylab = "counts")
      for(i in base::seq_len(10)){
        base::cat(base::paste0("Simulation ", i , "\n"))
        library(package_name)  # Replace 'package_name' with the actual name of the package that contains the 'getRandomSubsample' function

        random_subsample <- getRandomSubsample(cdr3_len_stratify = cdr3_len_stratify,
                                               vgene_stratify = vgene_stratify,
                                               refseqs_motif_region = refseqs_motif_region,
                                               motif_region = motif_region,
                                               motif_lengths_list = motif_lengths_list,
                                               ref_motif_lengths_id_list = ref_motif_lengths_id_list,
                                               motif_region_vgenes_list = motif_region_vgenes_list,
                                               ref_motif_vgenes_id_list = ref_motif_vgenes_id_list,
                                               ref_lengths_vgenes_list = ref_lengths_vgenes_list,
                                               lengths_vgenes_list = lengths_vgenes_list)
        counts <- base::rep(0,20)
        # Get minimum distance to nearest neighbor for every sequence in random reference subsample
        for(i in base::seq_along(random_subsample)){
          distance <- base::nchar(random_subsample[i])
          dis <- base::min(stringdist::stringdist(a = random_subsample[i],b = random_subsample[-i],method = "hamming",nthread = 1))
          if(dis < distance) distance <- dis
          
          counts[distance] <- counts[distance] + 1
        }
        
        graphics::barplot(counts, names.arg = base::seq_len(20), main = base::paste("Global convergence distribution - ", base::paste0("Simulation ", i), sep = ""), xlab = "Hamming distance", ylab = "counts")
        
      }
    }
    
    ### Compute motif frequency in sample set
    discovery <- find_motifs(seqs = motif_region,q = motif_length, discontinuous = discontinuous)
    
    ### Compute motif frequency in reference subsamples at sample set depth
    res <- foreach::foreach(i = base::seq_len(sim_depth), .inorder = FALSE) %dopar% {
      motif_sample <- getRandomSubsample(cdr3_len_stratify = cdr3_len_stratify,
                                         vgene_stratify = vgene_stratify,
                                         refseqs_motif_region = refseqs_motif_region,
                                         motif_region = motif_region,
                                         motif_lengths_list = motif_lengths_list,
                                         ref_motif_lengths_id_list = ref_motif_lengths_id_list,
                                         motif_region_vgenes_list = motif_region_vgenes_list,
                                         ref_motif_vgenes_id_list = ref_motif_vgenes_id_list,
                                         ref_lengths_vgenes_list = ref_lengths_vgenes_list,
                                         lengths_vgenes_list = lengths_vgenes_list)
      
      sim <- find_motifs(seqs = motif_sample,q = motif_length, discontinuous = discontinuous)
      sim <- dplyr::left_join(discovery,sim,"motif")
      sim$V1.y[base::is.na(sim$V1.y)] <- 0
      simy <- sim$V1.y
      simy
    }
    
    ### Adjust part 1 results
    # convert results into a data frame with iteration represented by the row and motif represented by the column
    res <- base::unlist(res)
    copies <- base::matrix(base::as.numeric(res), ncol = sim_depth, byrow = FALSE)
    part_1_res <- base::cbind(discovery$motif,discovery$V1,copies)
    part_1_res <- base::t(part_1_res)
    mot <- part_1_res[1,]
    part_1_res <- part_1_res[-1,]
    part_1_res <- base::matrix(base::as.numeric(part_1_res),nrow = sim_depth+1)
    base::colnames(part_1_res) <- mot
    part_1_res <- base::as.data.frame(part_1_res)
    base::rownames(part_1_res) <- base::c("Discovery",base::paste("sim",0:(sim_depth-1),sep = "-"))
    
    if(save_results == TRUE) utils::write.table(x = part_1_res,file = part_1_res_file_name,quote = FALSE,sep = "\t",row.names = TRUE)
    part_1_res <- base::as.data.frame(part_1_res)
    
    t1_e <- base::Sys.time()
    base::cat("Part 1 cpu time:",t1_e-t1,base::units(t1_e-t1),"\n\n")
    
    ##################################################################
    ##               Part 2: Identify enriched motifs               ##
    ##################################################################
    base::cat(base::paste("Part 2: Selecting motifs significantly more represented in the data compared to the reference DB. \n"))
    
    part_1_res_file_name <- base::paste0(result_folder,"kmer_resample_",sim_depth,"_log.txt")
    part_2_res_file_name <- base::paste0(result_folder,"kmer_resample_",sim_depth,"_minp",lcminp,"_ove",paste(lcminove, collapse = "_"),".txt")
    all_part_2_res_file_name <- base::paste0(result_folder,"kmer_resample_",sim_depth,"_all_motifs.txt")
    
    ### Take along the results of part 1
    # actual_reads = motif occurences in the sample set
    # sample_reads = motif occurences in the repeated random sampling
    sample_reads <- part_1_res
    actual_reads <- base::as.data.frame(sample_reads[1,])
    base::colnames(actual_reads) <- base::colnames(sample_reads)
    sample_reads <- base::as.data.frame(sample_reads[-1,])
    nam <- base::colnames(actual_reads)
    
    ### Prepare framework for part 2 results
    part_2_res <- base::matrix(ncol = 6,dimnames = base::c(base::list(""),base::list(base::c("Motif",	"counts",	"avgRef",	"topRef",	"OvE",	"p-value"))))
    part_2_res <- part_2_res[-1,]
    all_part_2_res <- part_2_res
    
    ### Get the minimum fold change for every motif depending on the motif length
    if(base::length(lcminove) == 1){
      temp_minove <- base::rep(lcminove, base::ncol(actual_reads))
    } else {
      temp_minove <- base::rep(0, base::ncol(actual_reads))
      
      # the number of informative positions in discontinuous motifs is reduced by one 
      nam_len <- base::nchar(nam)
      nam_len[base::grep(".", nam, fixed = TRUE)] <- nam_len[base::grep(".", nam, fixed = TRUE)]-1
      
      for(i in base::seq_along(motif_length)){
        temp_minove[nam_len == motif_length[i]] <- lcminove[i]
      }
    }
    
    ### Filter the significant motifs by the following criteria:
    # p-value < lcminp (p-value = probability of observing the motif in a random reference subsample at sample set depth at least as often as in the sample set)
    # fold change >= lcminove (lcminove can be dependent on the motif length; fold change of sample set occurence compared to repeated random subsample occurence)
    # sample set occurence >= kmer_mindepth
    for(i in base::seq_len(base::ncol(actual_reads))){
      p.v <- base::length(base::which(sample_reads[,i] >= actual_reads[,i])) / sim_depth
      
      m <- base::mean(sample_reads[,i])
      maximum <- base::max(sample_reads[,i])
      ove <- 0
      if(m > 0) {
        ove <- actual_reads[,i]/m
      } else {
        ove <- 1/(sim_depth*length(motif_region))
      }
      if(ove >= temp_minove[i] &&
         p.v < lcminp &&
         actual_reads[,i] >= kmer_mindepth){
        
        if(p.v == 0) p.v <- 1/sim_depth
        
        part_2_res <- base::rbind(part_2_res,
                                  base::c(nam[i],
                                          actual_reads[,i],
                                          base::round(m,digits = 2),
                                          maximum,
                                          base::round(ove,digits = 3),
                                          base::round(p.v,digits = 8)))
        
      }
      all_part_2_res <- base::rbind(all_part_2_res,
                                    base::c(nam[i],
                                            actual_reads[,i],
                                            base::round(m,digits = 2),
                                            maximum,
                                            base::round(ove,digits = 3),
                                            base::round(p.v,digits = 8)))
    }
    if(save_results == TRUE){
      utils::write.table(x = all_part_2_res,file = all_part_2_res_file_name,quote = FALSE,sep = "\t",row.names = FALSE)
      utils::write.table(x = part_2_res,file = part_2_res_file_name,quote = FALSE,sep = "\t",row.names = FALSE)
    }
    all_part_2_res <- base::as.data.frame(all_part_2_res)
    part_2_res <- base::as.data.frame(part_2_res)
    
    if(base::nrow(part_2_res) == 0){
      local_similarities <- FALSE
      base::cat("No significantly enriched motifs found in sample set.\n")
    } else {
      base::cat(base::nrow(part_2_res),"significantly enriched motifs found in sample set.\n")
    }
    
    t2_e <- base::Sys.time()
    base::cat("Part 2 cpu time:",t2_e-t1_e,base::units(t2_e-t1_e),"\n\n")
  } else {
    part_1_res <- base::c()
    all_part_2_res <- base::c()
    part_2_res <- base::c()
    
    base::cat(base::paste("\nLocal similarity search (part 1 and part 2) is skipped. \n\n"))
  }
  
  ##################################################################
  ##  Part 3: Identification of global similarities & clustering  ##
  ##################################################################
  
  if(local_similarities == FALSE){
    base::cat(base::paste("Part 3: Clustering the sequences based on global similarity. \n"))
    t2_e <- base::Sys.time()
  } else if(global_similarities == FALSE){
    base::cat(base::paste("Part 3: Clustering the sequences based on local similarity. \n"))
  } else {
    base::cat(base::paste("Part 3: Clustering the sequences based on local and global similarity. \n"))
  }
  
  part_2_res_file_name <- base::paste0(result_folder,"kmer_resample_",sim_depth,"_minp",lcminp,"_ove",paste(lcminove, collapse = "_"),".txt")
  part_3_res_file_name <- base::paste0(result_folder,"clone_network.txt")
  
  ### Prepare tuples of sequences and v genes (if given)
  if(accept_sequences_with_C_F_start_end) seqs <- base::grep(pattern = "^C.*F$",x = sequences$CDR3b,perl = TRUE,value = TRUE)
  temp_seqs <- base::c()
  temp_vgenes <- base::c()
  if(global_vgene == TRUE){
    temp_seqs <- seqs
    temp_vgenes <- sequences$TRBV[base::which(sequences$CDR3b %in% temp_seqs)]
  }
  seqs <- base::unique(seqs)
  seqs <- seqs[base::which(base::nchar(seqs) >= min_seq_length)]
  
  ### Reduce sample sequences to the motif region
  motif_region <- base::c()
  if(structboundaries) motif_region <- base::substr(x = seqs,start = boundary_size + 1 ,stop = base::nchar(seqs)-boundary_size) else motif_region <- seqs
  
  ### Set global convergency cutoff
  if(base::is.null(gccutoff)) if(base::length(seqs) < 125) gccutoff <- 2 else gccutoff <- 1
  
  clone_network_for_write <- base::c()

  
  if(global_similarities == TRUE){
    
    ### Identify global similarities
    res <- foreach::foreach(i = base::seq_along(seqs)) %dopar% {
      not_in_global_ids  <- base::c()
      global_con <- base::c()
      
      # Compute hamming distance between all sequences and select distances lower or equal to gccutoff
      dis <- stringdist::stringdist(a = motif_region[i],b = motif_region,method = "hamming",nthread = 1) #F: hamming Inf problem
      global_ids <- base::which(dis <= gccutoff ) #source of error : for short sequences there is a high chance that their hamming distance <=1 when we remove the boundries 3 at start and 3 at the end
      
      # If requested, restrict global similarities to shared v genes
      if(global_vgene == TRUE){
        act_vgenes <- temp_vgenes[temp_seqs == seqs[i]]
        global_ids_seqs <- seqs[global_ids]
        all_global_ids_seqs <- temp_seqs[temp_seqs %in% global_ids_seqs]
        global_ids <- global_ids[global_ids_seqs %in% all_global_ids_seqs[temp_vgenes[temp_seqs %in% global_ids_seqs] %in% act_vgenes]]
      }
      
      # If no neighbor present remind the ID, else create clone network
      if(base::length(global_ids) == 1) not_in_global_ids <- i
      global_ids <- global_ids[base::which(global_ids > i)]
      if(base::length(global_ids)>0){
        global_con <- base::cbind(base::rep(seqs[i],base::length(global_ids)),seqs[global_ids],base::rep("global",base::length(global_ids)))
      }
      base::list(not_in_global_ids = not_in_global_ids, global_con = global_con)
    }
    
    ### Build clone network
    not_in_global_ids <- base::c()
    global_con <- base::c()
    for(i in base::seq_along(res)){
      not_in_global_ids <- base::c(not_in_global_ids, res[[i]]$not_in_global_ids)
      global_con <- base::rbind(global_con, res[[i]]$global_con)
    }
    clone_network_for_write <- base::rbind(clone_network_for_write, global_con)
    
    ### Test randomness of global similarities
    if(accept_sequences_with_C_F_start_end == TRUE && global_vgene == FALSE && structboundaries == TRUE && boundary_size == 3 && !base::is.null(clone_network_for_write)){
      base::cat("Testing the randomness of the number of global similarities in the sample.\n")
      num_globals <- base::sum(clone_network_for_write[,3] == "global")
      base::cat("p-value: ", stats::pnorm(num_globals,
                                          9.786049e-06*base::length(motif_region)^2,
                                          1.421e-07*base::length(motif_region)^2 + 3.176e-03*base::length(motif_region), lower.tail = FALSE), "\n")
    }
  } else {
    base::cat(base::paste("Global similarity search is skipped. \n\n"))
    not_in_global_ids <- base::seq_along(seqs)
  }
  
  ### local similarity clustering
  in_local_ids <- base::c()
  
  if(local_similarities == TRUE){
    
    motifs <- part_2_res[,1]
    
    # For every motif identify the sequences containing the motif and add the connections to the clone network
    if(base::length(motifs) > 0){
      for(j in base::seq_along(motifs)){
        local_ids <- base::grep(pattern = motifs[j],x = motif_region,value = FALSE)
        if(base::length(local_ids) > 1){ #Draius15042020
          in_local_ids <- base::c(in_local_ids, local_ids)
          local_con <- base::t(utils::combn(x = seqs[local_ids],m = 2))
          local_con <- base::cbind(local_con, base::rep("local", base::nrow(local_con)))
          if(positional_motifs == TRUE){
            if(structboundaries == TRUE ){
              local_con <- local_con[stringr::str_locate(base::substr(local_con[,1], boundary_size+1, nchar(local_con[,1])-boundary_size), pattern = motifs[j])[,1] ==
                                     stringr::str_locate(base::substr(local_con[,2], boundary_size+1, nchar(local_con[,2])-boundary_size), pattern = motifs[j])[,1],]
            } else {
              local_con <- local_con[stringr::str_locate(local_con[,1], pattern = motifs[j])[,1] == stringr::str_locate(local_con[,2], pattern = motifs[j])[,1],]
            }
          }
          if(base::nrow(local_con) == 0) local_con <- base::c()
          clone_network_for_write <- base::rbind(clone_network_for_write, local_con)
        }
      }
    }
  }
  
  ### if tcrs from different donors are not allowed in one cluster, remove this connections
  if(public_tcrs == FALSE && patient.info == TRUE && !base::is.null(clone_network_for_write)){
    temp <- base::apply(clone_network_for_write, 1, function(x){
      if(base::any(sequences$patient[sequences$CDR3b == x[1]] %in% sequences$patient[sequences$CDR3b == x[2]])) TRUE else FALSE
    })
    clone_network_for_write <- clone_network_for_write[temp,]
    
    if(base::nrow(clone_network_for_write) == 0) clone_network_for_write <- base::c()
  }
  
  ### singletons (sequences without connection)
  singleton_ids <- base::setdiff(x = not_in_global_ids,y=in_local_ids)
  if(base::length(singleton_ids) > 0){
    clone_network_for_write <- base::rbind(clone_network_for_write,base::cbind(seqs[singleton_ids],
                                                                               seqs[singleton_ids],
                                                                               base::rep("singleton",base::length(singleton_ids))))
  }

  if(save_results == TRUE) utils::write.table(x = clone_network_for_write,file = part_3_res_file_name,quote = FALSE,sep = "\t",row.names = FALSE,col.names = FALSE)
  
  t3_e <- base::Sys.time()
  base::cat("Part 3 cpu time:",t3_e-t2_e,base::units(t3_e-t2_e),"\n\n\n")
  
  ##################################################################
  ##            Part 4: Finding the network components            ##
  ##################################################################
  part_4_file_name <- base::paste0(result_folder,"convergence_groups.txt")
  
  base::cat("Part 4: Finding the network components. \n")
  
  part_4_res <- base::c()
  part_4_res_list <- base::c()
  save_cluster_list_df <- base::c()
  if(!base::is.null(clone_network_for_write)){
    clone_network_for_write <- base::as.data.frame(clone_network_for_write)
    temp_clone_network_for_write <- clone_network_for_write
    temp_clone_network_for_write[] <- base::lapply(temp_clone_network_for_write, base::as.character)
    
    ### build temporarily a clone network with all tuple information (sequence, v gene, donor), to get all member tuples of the clusters
    x <- NULL
    temp_clone_network_for_write <- foreach::foreach(x = base::seq_len(base::nrow(temp_clone_network_for_write))) %dopar% {
      # get all tuple IDs with this sequence
      act_ids1 <- base::which(sequences$CDR3b == temp_clone_network_for_write[x,1])
      act_ids2 <- base::which(sequences$CDR3b == temp_clone_network_for_write[x,2])
      
      # First assume conenctions between all tuples
      comb_ids <- base::expand.grid(act_ids1, act_ids2)
      comb_ids <- base::unique(comb_ids)
      
      # get all tuple information with this sequence
      act_infos1 <- sequences[comb_ids[,1],]
      act_infos2 <- sequences[comb_ids[,2],]
      
      # If requested, exclude connections between tuples with different donors
      if(public_tcrs == FALSE && patient.info == TRUE){
        exclude_rows <- act_infos1$patient != act_infos2$patient
        act_infos1 <- act_infos1[!exclude_rows,]
        act_infos2 <- act_infos2[!exclude_rows,]
      }
      
      # If requested, exclude global connections between tuples with different v genes
      if(global_vgene == TRUE && vgene.info == TRUE && temp_clone_network_for_write[x,3] == "global" && base::nrow(act_infos1) > 0){
        exclude_rows <- act_infos1$TRBV != act_infos2$TRBV
        act_infos1 <- act_infos1[!exclude_rows,]
        act_infos2 <- act_infos2[!exclude_rows,]
      }
      
      if(base::nrow(act_infos1) > 0){
        # insert a very uggly seperator between the information that probably will never occur in any input
        act_infos1 <- base::do.call(base::paste, base::c(act_infos1, sep="$#$#$"))
        act_infos2 <- base::do.call(base::paste, base::c(act_infos2, sep="$#$#$"))
        
        var_ret <- base::data.frame(act_infos1, act_infos2)
        base::t(var_ret)
      } else return(c())
      
    }
    temp_clone_network_for_write <- base::data.frame(base::matrix(base::unlist(temp_clone_network_for_write), ncol = 2, byrow = TRUE))
    temp_clone_network_for_write[] <- base::lapply(temp_clone_network_for_write, base::as.character)
    
    ### Receive the network from the clone network using the igraph-package
    gr <- igraph::graph_from_edgelist(el = base::as.matrix(temp_clone_network_for_write[,base::c(1,2)]),directed = FALSE)
    cm <- igraph::components(gr)
    
    ### Extract the cluster compositions of the created network
    all_leaders <- base::c()
    for(i in base::seq_len(cm$no)){
      # extract the member tuples of the current cluster
      members_id <- base::names(cm$membership)[base::which(cm$membership == i)]
      members_id <- base::unique(members_id)
      members_id <- base::data.frame(base::matrix(base::unlist(base::strsplit(members_id, split = "$#$#$", fixed = TRUE)), ncol = base::ncol(sequences), byrow = TRUE),
                                     stringsAsFactors = FALSE)
      base::colnames(members_id) <- base::colnames(sequences)
      
      # determine the cluster size (= number of unique CDR3b sequences)
      members <- base::sort(base::unique(members_id$CDR3b))
      csize <- base::length(members)
      
      # get the representative leader tag of the cluster
      leader <- base::paste("CRG",members[1],sep = "-")
      while_counter <- 1
      temp_leader <- leader
      while(temp_leader %in% all_leaders){
        temp_leader <- base::paste0(leader, "-", while_counter)
        while_counter <- while_counter + 1
      }
      leader <- temp_leader
      all_leaders <- base::c(all_leaders, leader)
      
      if(i == 1){
        part_4_res <- base::data.frame(cluster_size = csize, tag = leader,members = base::paste(members,collapse = " "),
                                       stringsAsFactors = FALSE)
      } else {
        part_4_res <- base::rbind(part_4_res, base::data.frame(cluster_size = csize, tag = leader,members = base::paste(members,collapse = " "),
                                                               stringsAsFactors = FALSE))
      }
      cluster_info <- members_id
      part_4_res_list[[leader]] <- cluster_info
    }
    
    ### exclude clusters with less size than cluster_min_size
    eliminate_ids <- base::which(part_4_res$cluster_size < cluster_min_size)
    if(base::length(eliminate_ids) > 0){
      part_4_res <- part_4_res[-eliminate_ids,]
      part_4_res_list <- part_4_res_list[-eliminate_ids]
    }
    
    if(base::nrow(part_4_res) > 0){
      ### save part_4_res_list
      save_cluster_list_df <- foreach::foreach(i = base::seq_along(part_4_res_list), .combine = "rbind") %dopar% {
        temp <- part_4_res_list[[i]]
        temp <- base::cbind(base::data.frame(tag = base::rep(base::names(part_4_res_list)[i], base::nrow(temp))), temp)
      }
    } else {
      part_4_res <- base::c()
      part_4_res_list <- base::c()
      save_cluster_list_df <- base::c()
    }
  }
  
  if(save_results == TRUE) utils::write.table(x = save_cluster_list_df,file = base::paste0(result_folder, "cluster_member_details.txt"),quote = FALSE,sep = "\t",row.names = FALSE,col.names = TRUE)
  
  t4_e <- base::Sys.time()
  base::cat("Part 4 cpu time:",t4_e-t3_e,base::units(t4_e-t3_e), "\n\n\n")
  
  doParallel::stopImplicitCluster()
  
  #################################################################
  ##                   Part 5: Cluster scoring                   ##
  #################################################################
  
  base::cat("Part 5: Scoring all convergence groups with equal or more than ", cluster_min_size, " members. \n")
  
  if(!base::is.null(part_4_res_list)){
    scoring_res <- cluster_scoring(cluster_list = part_4_res_list,
                                               cdr3_sequences = cdr3_sequences,
                                               refdb_beta = refdb_beta,
                                               v_usage_freq = v_usage_freq,
                                               cdr3_length_freq = cdr3_length_freq,
                                               ref_cluster_size = ref_cluster_size,
                                               sim_depth = sim_depth,
                                               gliph_version = 1,
                                               hla_cutoff = hla_cutoff,
                                               n_cores = no_cores)
    
    part_4_res <- base::cbind(part_4_res, scoring_res)
    # some reordering of columns
    part_4_res <- part_4_res[, base::c(base::colnames(part_4_res)[base::colnames(part_4_res) != "members"], "members")]
  }
  
  if(save_results == TRUE){
    utils::write.table(x = part_4_res, file = part_4_file_name ,quote = FALSE,sep = "\t",row.names = FALSE,col.names = TRUE)
  }
  
  t5_e <- base::Sys.time()
  base::cat("Part 5 cpu time:",t5_e-t4_e, base::units(t5_e-t4_e),"\n\n")
  
  ##################################################################
  ##                      Output preparation                      ##
  ##################################################################
  
  output <- base::list(sample_log = NULL,
                       motif_enrichment = NULL,
                       connections = NULL,
                       cluster_properties = NULL,
                       cluster_list = NULL,
                       parameters = NULL)
  
  output$sample_log <-  part_1_res
  
  # set all theoretical numeric values (actually characters) to numeric values
  if(base::is.data.frame(part_2_res)){
    for(i in base::seq_len(base::ncol(part_2_res))){
      if(base::suppressWarnings(base::any(base::is.na(base::as.numeric(part_2_res[,i])))) == FALSE) part_2_res[,i] <- base::as.numeric(part_2_res[,i])
    }
  }
  if(base::is.data.frame(all_part_2_res)){
    for(i in base::seq_len(base::ncol(all_part_2_res))){
      if(base::suppressWarnings(base::any(base::is.na(base::as.numeric(all_part_2_res[,i])))) == FALSE) all_part_2_res[,i] <- base::as.numeric(all_part_2_res[,i])
    }
  }
  output$motif_enrichment <- base::list(selected_motifs = part_2_res,all_motifs = all_part_2_res)
  
  clone_network_for_write[] <- base::lapply(clone_network_for_write, as.character)
  output$connections <- clone_network_for_write
  
  output$cluster_properties = part_4_res
  
  # set all theoretical numeric values (actually characters) to numeric values
  if(base::is.list(part_4_res_list)){
    for(i in base::seq_along(part_4_res_list)){
      if(base::is.data.frame(part_4_res_list[[i]])){
        for(j in base::seq_len(base::ncol(part_4_res_list[[i]]))){
          if(base::suppressWarnings(base::any(base::is.na(base::as.numeric(part_4_res_list[[i]][,j])))) == FALSE) part_4_res_list[[i]][,j] <- base::as.numeric(part_4_res_list[[i]][,j])
        }
      }
      
    }
  }
  output$cluster_list <- part_4_res_list
  
  output$parameters <- base::list(gliph_version = 1,
                            result_folder = result_folder,
                            ref_cluster_size = ref_cluster_size,
                            sim_depth = sim_depth,
                            lcminp = lcminp,
                            lcminove = lcminove,
                            kmer_mindepth = kmer_mindepth,
                            accept_sequences_with_C_F_start_end = accept_sequences_with_C_F_start_end,
                            min_seq_length = min_seq_length,
                            gccutoff = gccutoff,
                            structboundaries = structboundaries,
                            boundary_size = boundary_size,
                            motif_length = motif_length,
                            discontinuous = discontinuous,
                            make_depth_fig = make_depth_fig,
                            local_similarities = local_similarities,
                            global_similarities = global_similarities,
                            global_vgene = global_vgene,
                            positional_motifs = positional_motifs,
                            cdr3_len_stratify = cdr3_len_stratify,
                            vgene_stratify = vgene_stratify,
                            public_tcrs = public_tcrs,
                            cluster_min_size = cluster_min_size,
                            hla_cutoff = hla_cutoff,
                            n_cores = n_cores)
  
  ### save parameters
  paras <- base::c()
  for(i in base::seq_along(output$parameters)){
    temp_paras <- base::data.frame(base::names(output$parameters)[i], base::paste0(output$parameters[[i]], collapse = ","), stringsAsFactors = FALSE)
    if(i == 1) paras <- temp_paras else paras <- base::rbind(paras, temp_paras)
  }
  if(save_results == TRUE) utils::write.table(x = paras,file = base::paste0(result_folder, "parameter.txt"),quote = FALSE,sep = "\t",row.names = FALSE,col.names = FALSE)
  
  t2 <- base::Sys.time()
  dt <- (t2-t1)
  base::cat("Total time = ",dt, base::units(dt),"\n")
  
  if(save_results == TRUE) base::cat("Output: results are stored in ", result_folder, "\n")
  
  ### Closing time!
  base::return(output)
}